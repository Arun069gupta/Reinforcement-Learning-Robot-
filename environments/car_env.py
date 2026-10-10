import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from environments.track import RacingTrack


class CarEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(self):

        super().__init__()

        # ==========================================================
        # MUJOCO MODEL
        # ==========================================================

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models",
            "car.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(
            model_path
        )

        self.data = mujoco.MjData(
            self.model
        )

        # ==========================================================
        # TRACK
        # ==========================================================

        self.track = RacingTrack()

        # ==========================================================
        # ACTION SPACE
        # ==========================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ==========================================================
        # OBSERVATION SPACE
        #
        # 0  x
        # 1  y
        # 2  vx
        # 3  vy
        # 4  sin(yaw)
        # 5  cos(yaw)
        # 6  distance from centerline
        # 7  progress
        # 8  waypoint direction X
        # 9  waypoint direction Y
        # 10 sin(heading error)
        # 11 cos(heading error)
        # 12 sin(lookahead error)
        # 13 cos(lookahead error)
        # 14 turn strength
        # 15 turn direction
        # 16 recovery direction X
        # 17 recovery direction Y
        # 18 recovery distance
        # ==========================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19,),
            dtype=np.float32
        )

        # ==========================================================
        # EPISODE
        # ==========================================================

        self.max_steps = 5000
        self.step_count = 0

        # ==========================================================
        # CURRENT ATTEMPT
        # ==========================================================

        self.attempt_number = 0
        self.current_progress = 0.0

        # ==========================================================
        # GLOBAL LEARNING MEMORY
        # ==========================================================

        self.best_progress = 0.0

        self.best_reward = -np.inf

        self.best_position = np.zeros(
            2,
            dtype=np.float32
        )

        # ==========================================================
        # FAILED ROUTES
        # ==========================================================

        self.failed_routes = []

        # ==========================================================
        # LAST SAFE POSITION
        # ==========================================================

        self.last_safe_position = np.zeros(
            2,
            dtype=np.float32
        )

        self.recovery_distance = 0.0
        self.previous_recovery_distance = 0.0

        # ==========================================================
        # DISTANCE MEMORY
        # ==========================================================

        self.previous_centerline_distance = 0.0

        # ==========================================================
        # PREVIOUS STATE
        # ==========================================================

        self.previous_progress = 0.0

        self.previous_position = np.zeros(
            2,
            dtype=np.float32
        )

        self.previous_action = np.zeros(
            2,
            dtype=np.float32
        )

        # ==========================================================
        # FAILURE SETTINGS
        # ==========================================================

        self.failure_distance = 1.50
        self.max_track_distance = 2.0

        # ==========================================================
        # FAILURE MEMORY
        # ==========================================================

        self.failure_progress_window = 0.025
        self.failure_position_radius = 0.75

        # ==========================================================
        # REWARD SETTINGS
        # ==========================================================

        self.progress_reward_scale = 100.0

        self.forward_reward_scale = 0.5

        self.reverse_penalty_scale = 8.0

        # ==========================================================
        # START POSITION
        # ==========================================================

        self.start_position = np.array(
            [-4.5, -4.25],
            dtype=np.float32
        )

        self.start_z = 0.32

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        # ==========================================================
        # RESET MUJOCO
        # ==========================================================

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # ==========================================================
        # START POSITION
        # ==========================================================

        self.data.qpos[0] = self.start_position[0]
        self.data.qpos[1] = self.start_position[1]
        self.data.qpos[2] = self.start_z

        # ==========================================================
        # YAW = 0
        # ==========================================================

        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        # ==========================================================
        # ZERO VELOCITY
        # ==========================================================

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        # ==========================================================
        # INITIAL TRACK STATE
        # ==========================================================

        position = self.data.qpos[0:2].copy()

        (
            progress,
            segment,
            distance
        ) = self.track.get_track_info(
            position
        )

        # ==========================================================
        # CURRENT ATTEMPT RESET
        # ==========================================================

        self.step_count = 0

        self.current_progress = progress

        self.previous_progress = progress

        self.previous_position = position.copy()

        self.previous_action[:] = 0.0

        # ==========================================================
        # RECOVERY RESET
        # ==========================================================

        self.last_safe_position = position.copy()

        self.recovery_distance = 0.0

        self.previous_recovery_distance = 0.0

        self.previous_centerline_distance = distance

        # ==========================================================
        # OBSERVATION
        # ==========================================================

        observation = self._get_observation()

        return observation, {}

    # ==============================================================
    # STEP
    # ==============================================================

    def step(self, action):

        # ==========================================================
        # ACTION
        # ==========================================================

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # ==========================================================
        # EXTRACT MOTOR COMMANDS
        # ==========================================================

        left_action = float(action[0])
        right_action = float(action[1])

        # ==========================================================
        # STABLE WHEEL DRIVE


        self.data.ctrl[0] = left_action
        self.data.ctrl[1] = right_action
        for _ in range(50):
            mujoco.mj_step(
                self.model,
                self.data
            )

        # ==========================================================

        for physics_step in range(50):

            if physics_step % 2 == 0:

                # --------------------------------------------------
                # LEFT WHEEL PULSE
                # --------------------------------------------------

                self.data.ctrl[0] = left_action

                self.data.ctrl[1] = (
                    right_action * 0.25
                )

            else:

                # --------------------------------------------------
                # RIGHT WHEEL PULSE
                # --------------------------------------------------

                self.data.ctrl[0] = (
                    left_action * 0.25
                )

                self.data.ctrl[1] = right_action

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.step_count += 1

        # ==========================================================
        # POSITION
        # ==========================================================

        position = self.data.qpos[0:2].copy()

        # ==========================================================
        # TRACK INFORMATION
        # ==========================================================

        (
            progress,
            segment,
            distance
        ) = self.track.get_track_info(
            position
        )

        self.current_progress = progress

        # ==========================================================
        # TRACK DIRECTION
        # ==========================================================

        track_direction = (
            self.track.get_track_direction(
                segment
            )
        )

        # ==========================================================
        # VELOCITY
        # ==========================================================

        velocity = self.data.qvel[0:2].copy()

        forward_track_velocity = np.dot(
            velocity,
            track_direction
        )

        # ==========================================================
        # PROGRESS CHANGE
        # ==========================================================

        progress_change = (
            progress -
            self.previous_progress
        )

        # ==========================================================
        # LAP CROSSING
        # ==========================================================

        if progress_change < -0.5:

            progress_change += 1.0

        progress_change = np.clip(
            progress_change,
            -0.05,
            0.05
        )

        # ==========================================================
        # SAFE LINE
        # ==========================================================

        safe_half_width = (
            self.track.safe_half_width
        )

        inside_safe_line = (
            distance <= safe_half_width
        )

        # ==========================================================
        # UPDATE LAST SAFE POSITION
        # ==========================================================

        if inside_safe_line:

            self.last_safe_position = (
                position.copy()
            )

            self.recovery_distance = 0.0

            self.previous_recovery_distance = 0.0

        else:

            recovery_vector = (
                self.last_safe_position -
                position
            )

            self.recovery_distance = np.linalg.norm(
                recovery_vector
            )

        # ==========================================================
        # PROGRESS REWARD
        # ==========================================================

        progress_reward = (
            progress_change *
            self.progress_reward_scale
        )

        # ==========================================================
        # FORWARD REWARD
        # ==========================================================

        forward_reward = (
            np.clip(
                forward_track_velocity,
                0.0,
                1.0
            )
            *
            self.forward_reward_scale
        )

        # ==========================================================
        # BACKWARD PENALTY
        # ==========================================================

        reverse_penalty = 0.0

        if forward_track_velocity < -0.02:

            reverse_penalty = (
                abs(forward_track_velocity)
                *
                self.reverse_penalty_scale
            )

        # ==========================================================
        # SAFE LINE PENALTY
        # ==========================================================

        distance_outside = max(
            0.0,
            distance - safe_half_width
        )

        safe_line_penalty = (
            distance_outside ** 2
        ) * 30.0

        # ==========================================================
        # CENTERLINE RECOVERY
        # ==========================================================

        centerline_recovery_delta = (
            self.previous_centerline_distance -
            distance
        )

        centerline_recovery_reward = 0.0

        if not inside_safe_line:

            centerline_recovery_reward = (
                centerline_recovery_delta *
                30.0
            )

        self.previous_centerline_distance = distance

        # ==========================================================
        # PREVIOUS SAFE POSITION RECOVERY
        # ==========================================================

        recovery_reward = 0.0

        if not inside_safe_line:

            recovery_delta = (
                self.previous_recovery_distance -
                self.recovery_distance
            )

            recovery_reward = (
                recovery_delta *
                10.0
            )

            self.previous_recovery_distance = (
                self.recovery_distance
            )

        # ==========================================================
        # KNOWN FAILURE DETECTION
        # ==========================================================

        known_failure = False

        closest_failure = None

        closest_failure_distance = float(
            "inf"
        )

        for failure in self.failed_routes:

            progress_difference = abs(
                progress -
                failure["progress"]
            )

            progress_difference = min(
                progress_difference,
                1.0 - progress_difference
            )

            if (
                progress_difference
                <=
                self.failure_progress_window
            ):

                position_difference = np.linalg.norm(
                    position -
                    failure["position"]
                )

                if (
                    position_difference
                    <=
                    self.failure_position_radius
                ):

                    known_failure = True

                    if (
                        position_difference
                        <
                        closest_failure_distance
                    ):

                        closest_failure = failure

                        closest_failure_distance = (
                            position_difference
                        )

        # ==========================================================
        # ALTERNATE ROUTE REWARD
        # ==========================================================

        alternate_route_reward = 0.0

        if (
            known_failure
            and
            inside_safe_line
            and
            closest_failure is not None
        ):

            route_difference = np.linalg.norm(
                position -
                closest_failure["position"]
            )

            alternate_route_reward = np.clip(
                route_difference,
                0.0,
                0.75
            ) * 3.0

        # ==========================================================
        # FRONTIER REWARD
        # ==========================================================

        frontier_reward = 0.0

        if progress > self.best_progress - 0.02:

            if forward_track_velocity > 0.03:

                frontier_reward = 2.0

        # ==========================================================
        # STAGNATION PENALTY
        # ==========================================================

        stagnation_penalty = 0.0

        if abs(forward_track_velocity) < 0.01:

            stagnation_penalty = 1.0

        # ==========================================================
        # TOTAL REWARD
        # ==========================================================

        reward = (

            progress_reward

            + forward_reward

            + centerline_recovery_reward

            + recovery_reward

            + alternate_route_reward

            + frontier_reward

            - safe_line_penalty

            - reverse_penalty

            - stagnation_penalty
        )

        # ==========================================================
        # UPDATE BEST PROGRESS
        # ==========================================================

        if (
            progress > self.best_progress
            and
            inside_safe_line
        ):

            self.best_progress = progress

            self.best_position = (
                position.copy()
            )

            self.best_reward = max(
                self.best_reward,
                reward
            )

        # ==========================================================
        # WALL COLLISION
        # ==========================================================

        wall_collision = (
            self._check_wall_collision()
        )

        # ==========================================================
        # SERIOUS FAILURE
        # ==========================================================

        serious_off_track = (
            distance >= self.failure_distance
        )

        failure = (
            wall_collision
            or
            serious_off_track
        )

        # ==========================================================
        # REGENERATION
        # ==========================================================

        regenerated = False

        failure_progress = progress

        if failure:

            # ------------------------------------------------------
            # SAVE FAILED ROUTE
            # ------------------------------------------------------

            self.failed_routes.append(
                {
                    "progress": float(
                        failure_progress
                    ),

                    "position": (
                        position.copy()
                    ),

                    "action": (
                        action.copy()
                    )
                }
            )

            # ------------------------------------------------------
            # KEEP MEMORY MANAGEABLE
            # ------------------------------------------------------

            if len(self.failed_routes) > 50:

                self.failed_routes.pop(0)

            # ------------------------------------------------------
            # FAILURE PENALTY
            # ------------------------------------------------------

            reward -= 50.0

            # ------------------------------------------------------
            # REGENERATE
            # ------------------------------------------------------

            self._regenerate()

            regenerated = True

        # ==========================================================
        # TIME LIMIT
        # ==========================================================

        terminated = False

        truncated = False

        if self.step_count >= self.max_steps:

            truncated = True

        # ==========================================================
        # UPDATE PREVIOUS STATE
        # ==========================================================

        self.previous_progress = progress

        self.previous_position = (
            position.copy()
        )

        self.previous_action = (
            action.copy()
        )

        # ==========================================================
        # OBSERVATION
        # ==========================================================

        observation = self._get_observation()

        # ==========================================================
        # INFO
        # ==========================================================

        info = {

            "progress": float(
                progress
            ),

            "best_progress": float(
                self.best_progress
            ),

            "segment": int(
                segment
            ),

            "distance": float(
                distance
            ),

            "forward_velocity": float(
                forward_track_velocity
            ),

            "safe_line": bool(
                inside_safe_line
            ),

            "recovery_distance": float(
                self.recovery_distance
            ),

            "failed_routes": int(
                len(self.failed_routes)
            ),

            "attempt": int(
                self.attempt_number
            ),

            "wall_collision": bool(
                wall_collision
            ),

            "regenerated": bool(
                regenerated
            ),

            "failure_progress": float(
                failure_progress
            ),

            "alternate_route_reward": float(
                alternate_route_reward
            ),

            "frontier_reward": float(
                frontier_reward
            ),

            "reverse_penalty": float(
                reverse_penalty
            )
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    # ==============================================================
    # REGENERATE
    # ==============================================================

    def _regenerate(self):

        # ==========================================================
        # INCREASE ATTEMPT NUMBER
        # ==========================================================

        self.attempt_number += 1

        # ==========================================================
        # RESET POSITION
        # ==========================================================

        self.data.qpos[0] = (
            self.start_position[0]
        )

        self.data.qpos[1] = (
            self.start_position[1]
        )

        self.data.qpos[2] = (
            self.start_z
        )

        # ==========================================================
        # RESET YAW
        # ==========================================================

        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        # ==========================================================
        # RESET VELOCITY
        # ==========================================================

        self.data.qvel[:] = 0.0

        self.data.ctrl[:] = 0.0

        # ==========================================================
        # UPDATE MUJOCO
        # ==========================================================

        mujoco.mj_forward(
            self.model,
            self.data
        )

        # ==========================================================
        # RESET CURRENT ATTEMPT STATE
        # ==========================================================

        position = self.data.qpos[0:2].copy()

        (
            progress,
            segment,
            distance
        ) = self.track.get_track_info(
            position
        )

        self.previous_progress = progress

        self.current_progress = progress

        self.previous_position = (
            position.copy()
        )

        # ==========================================================
        # RESET RECOVERY
        # ==========================================================

        self.last_safe_position = (
            position.copy()
        )

        self.recovery_distance = 0.0

        self.previous_recovery_distance = 0.0

        self.previous_centerline_distance = (
            distance
        )

        # ==========================================================
        # IMPORTANT
        #
        # best_progress IS NOT RESET.
        #
        # failed_routes IS NOT RESET.
        #
        # This allows later attempts to remember previous
        # experience.
        # ==========================================================

    # ==============================================================
    # WALL COLLISION DETECTION
    # ==============================================================

    def _check_wall_collision(self):

        for i in range(
            self.data.ncon
        ):

            contact = self.data.contact[i]

            geom1 = int(
                contact.geom1
            )

            geom2 = int(
                contact.geom2
            )

            name1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom1
            )

            name2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom2
            )

            if name1 is None:

                name1 = ""

            if name2 is None:

                name2 = ""

            if (
                "wall" in name1
                or
                "wall" in name2
            ):

                return True

        return False

    # ==============================================================
    # OBSERVATION
    # ==============================================================

    def _get_observation(self):

        # ==========================================================
        # POSITION
        # ==========================================================

        x = float(
            self.data.qpos[0]
        )

        y = float(
            self.data.qpos[1]
        )

        position = np.array(
            [x, y],
            dtype=np.float32
        )

        # ==========================================================
        # VELOCITY
        # ==========================================================

        vx = float(
            self.data.qvel[0]
        )

        vy = float(
            self.data.qvel[1]
        )

        # ==========================================================
        # YAW
        # ==========================================================

        quat = self.data.qpos[3:7]

        w = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]

        yaw = np.arctan2(
            2.0 * (
                w * qz +
                qx * qy
            ),
            1.0 -
            2.0 * (
                qy * qy +
                qz * qz
            )
        )

        # ==========================================================
        # TRACK INFO
        # ==========================================================

        (
            progress,
            segment,
            distance
        ) = self.track.get_track_info(
            position
        )

        # ==========================================================
        # NEXT WAYPOINT
        # ==========================================================

        next_waypoint = (
            self.track.get_next_waypoint(
                segment
            )
        )

        waypoint_vector = (
            next_waypoint -
            position
        )

        waypoint_distance = np.linalg.norm(
            waypoint_vector
        )

        if waypoint_distance > 1e-8:

            waypoint_direction = (
                waypoint_vector /
                waypoint_distance
            )

        else:

            waypoint_direction = np.array(
                [1.0, 0.0],
                dtype=np.float32
            )

        # ==========================================================
        # TRACK DIRECTION
        # ==========================================================

        track_direction = (
            self.track.get_track_direction(
                segment
            )
        )

        track_angle = np.arctan2(
            track_direction[1],
            track_direction[0]
        )

        # ==========================================================
        # HEADING ERROR
        # ==========================================================

        heading_error = (
            track_angle -
            yaw
        )

        heading_error = np.arctan2(
            np.sin(heading_error),
            np.cos(heading_error)
        )

        # ==========================================================
        # LOOKAHEAD
        # ==========================================================

        lookahead_segment = (
            segment + 5
        ) % self.track.num_waypoints

        lookahead_point = (
            self.track.get_centerline_point(
                lookahead_segment
            )
        )

        lookahead_vector = (
            lookahead_point -
            position
        )

        lookahead_distance = np.linalg.norm(
            lookahead_vector
        )

        if lookahead_distance > 1e-8:

            lookahead_direction = (
                lookahead_vector /
                lookahead_distance
            )

        else:

            lookahead_direction = (
                track_direction
            )

        lookahead_angle = np.arctan2(
            lookahead_direction[1],
            lookahead_direction[0]
        )

        # ==========================================================
        # LOOKAHEAD HEADING ERROR
        # ==========================================================

        lookahead_heading_error = (
            lookahead_angle -
            yaw
        )

        lookahead_heading_error = np.arctan2(
            np.sin(
                lookahead_heading_error
            ),
            np.cos(
                lookahead_heading_error
            )
        )

        # ==========================================================
        # TURN STRENGTH
        # ==========================================================

        angle_difference = (
            lookahead_angle -
            track_angle
        )

        angle_difference = np.arctan2(
            np.sin(angle_difference),
            np.cos(angle_difference)
        )

        signed_turn_strength = np.sin(
            angle_difference
        )

        # ==========================================================
        # TURN DIRECTION
        # ==========================================================

        upcoming_turn_direction = np.sign(
            signed_turn_strength
        )

        # ==========================================================
        # RECOVERY VECTOR
        # ==========================================================

        recovery_vector = (
            self.last_safe_position -
            position
        )

        recovery_distance = np.linalg.norm(
            recovery_vector
        )

        if recovery_distance > 1e-8:

            recovery_direction = (
                recovery_vector /
                recovery_distance
            )

        else:

            recovery_direction = np.array(
                [0.0, 0.0],
                dtype=np.float32
            )

        # ==========================================================
        # OBSERVATION
        # ==========================================================

        observation = np.array(

            [

                x,
                y,

                vx,
                vy,

                np.sin(yaw),
                np.cos(yaw),

                distance,

                progress,

                waypoint_direction[0],
                waypoint_direction[1],

                np.sin(
                    heading_error
                ),

                np.cos(
                    heading_error
                ),

                np.sin(
                    lookahead_heading_error
                ),

                np.cos(
                    lookahead_heading_error
                ),

                signed_turn_strength,

                upcoming_turn_direction,

                recovery_direction[0],
                recovery_direction[1],

                recovery_distance

            ],

            dtype=np.float32

        )

        return observation

    # ==============================================================
    # CLOSE
    # ==============================================================

    def close(self):

        pass