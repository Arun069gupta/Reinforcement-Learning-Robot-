
import numpy as np
import mujoco
import gymnasium as gym

from ghost_training.ghost_car_env import GhostCarEnv


class SensorGhostCarEnv(GhostCarEnv):
    """
    Sensor-enabled MuJoCo autonomous racing environment.

    Observation:
        0-18  : Original CarEnv navigation observations
        19-29 : 11 virtual LiDAR readings
        30-35 : 6 IMU-like readings
        36-38 : 3 velocity readings

    Total observations: 39

    PPO action:
        action[0] = steering, from -1 (left) to +1 (right)
        action[1] = throttle, from -1 to +1

    The controller converts steering/throttle into left/right
    motor commands. It biases the throttle toward forward movement.
    """

    NUM_LIDAR_RAYS = 11
    LIDAR_MAX_RANGE = 10.0

    LIDAR_ANGLES = np.deg2rad(
        np.linspace(-90.0, 90.0, NUM_LIDAR_RAYS)
    )

    SPEED_SCALE = 20.0
    GYRO_SCALE = 10.0
    ACCEL_SCALE = 20.0

    # 0 = use previous action only; 1 = use new action immediately.
    ACTION_SMOOTHING = 0.35

    # Must match the number of physics steps in the parent CarEnv.
    PHYSICS_STEPS_PER_ACTION = 50

    # Vehicle control settings.
    MIN_THROTTLE = 0.35
    MAX_THROTTLE = 1.0
    STEERING_STRENGTH = 0.45

    def __init__(self, ghost_id=0):
        super().__init__(ghost_id=ghost_id)

        # Original observations (19) + LiDAR (11)
        # + IMU (6) + velocity (3) = 39.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(39,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.car_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "car",
        )

        if self.car_body_id < 0:
            raise RuntimeError(
                "Could not find body 'car' in models/car.xml."
            )

        self.sensor_dt = (
            float(self.model.opt.timestep)
            * self.PHYSICS_STEPS_PER_ACTION
        )

        if self.sensor_dt <= 0:
            raise RuntimeError("MuJoCo timestep must be positive.")

        self.previous_sensor_velocity = np.zeros(3, dtype=np.float64)

        # Smoothed policy action: [steering, throttle].
        self.previous_action_smoothed = np.zeros(
            2, dtype=np.float32
        )

        self.last_lidar_readings = np.ones(
            self.NUM_LIDAR_RAYS, dtype=np.float32
        )
        self.last_imu_readings = np.zeros(6, dtype=np.float32)
        self.last_velocity_readings = np.zeros(3, dtype=np.float32)

    # ----------------------------------------------------------
    # RESET
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.previous_sensor_velocity[:] = 0.0
        self.previous_action_smoothed[:] = 0.0

        self.last_lidar_readings[:] = 1.0
        self.last_imu_readings[:] = 0.0
        self.last_velocity_readings[:] = 0.0

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        self.previous_sensor_velocity = self.data.qvel[0:3].copy()

        # Rebuild observation after resetting the sensor baseline.
        observation = self._get_observation()
        info["ghost_id"] = self.ghost_id

        return observation, info

    # ----------------------------------------------------------
    # VEHICLE ORIENTATION
    # ----------------------------------------------------------

    def get_yaw(self):
        """Get yaw in radians from the free-joint quaternion."""

        w, x, y, z = self.data.qpos[3:7]

        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        return float(yaw)

    # ----------------------------------------------------------
    # VIRTUAL LIDAR
    # ----------------------------------------------------------

    def get_lidar_readings(self):
        """
        Cast 11 rays from left to right across a 180-degree field
        of view. Readings are normalized from 0 to 1.
        """

        yaw = self.get_yaw()

        origin = self.data.qpos[0:3].copy()
        origin[0] += 0.45 * np.cos(yaw)
        origin[1] += 0.45 * np.sin(yaw)
        origin[2] += 0.08

        readings = np.ones(
            self.NUM_LIDAR_RAYS,
            dtype=np.float32,
        )

        for i, relative_angle in enumerate(self.LIDAR_ANGLES):
            world_angle = yaw + float(relative_angle)

            direction = np.array(
                [
                    np.cos(world_angle),
                    np.sin(world_angle),
                    0.0,
                ],
                dtype=np.float64,
            )

            hit_geom = np.array([-1], dtype=np.int32)

            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                None,
                1,
                self.car_body_id,
                hit_geom,
            )

            if distance < 0.0 or not np.isfinite(distance):
                distance = self.LIDAR_MAX_RANGE

            distance = min(
                float(distance),
                self.LIDAR_MAX_RANGE,
            )

            readings[i] = np.clip(
                distance / self.LIDAR_MAX_RANGE,
                0.0,
                1.0,
            )

        self.last_lidar_readings = readings
        return readings.copy()

    # ----------------------------------------------------------
    # VIRTUAL IMU
    # ----------------------------------------------------------

    def get_imu_readings(self):
        """
        Output order:
            0: yaw angular velocity
            1: roll angular velocity
            2: pitch angular velocity
            3: longitudinal acceleration
            4: lateral acceleration
            5: vertical specific force

        Acceleration is estimated from velocity differences.
        """

        yaw = self.get_yaw()

        angular_velocity = self.data.qvel[3:6].copy()
        current_velocity = self.data.qvel[0:3].copy()

        acceleration_world = (
            current_velocity - self.previous_sensor_velocity
        ) / max(self.sensor_dt, 1e-8)

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        longitudinal_acceleration = (
            cos_yaw * acceleration_world[0]
            + sin_yaw * acceleration_world[1]
        )

        lateral_acceleration = (
            -sin_yaw * acceleration_world[0]
            + cos_yaw * acceleration_world[1]
        )

        vertical_specific_force = (
            acceleration_world[2] + 9.81
        )

        imu = np.array(
            [
                angular_velocity[2] / self.GYRO_SCALE,
                angular_velocity[0] / self.GYRO_SCALE,
                angular_velocity[1] / self.GYRO_SCALE,
                longitudinal_acceleration / self.ACCEL_SCALE,
                lateral_acceleration / self.ACCEL_SCALE,
                vertical_specific_force / self.ACCEL_SCALE,
            ],
            dtype=np.float32,
        )

        imu = np.nan_to_num(
            imu,
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        )

        self.last_imu_readings = np.clip(
            imu, -5.0, 5.0
        ).astype(np.float32)

        return self.last_imu_readings.copy()

    # ----------------------------------------------------------
    # VELOCITY
    # ----------------------------------------------------------

    def get_velocity_readings(self):
        """
        Output order:
            0: forward velocity
            1: lateral velocity
            2: total horizontal speed

        Values are normalized for the neural network.
        """

        yaw = self.get_yaw()

        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        forward_velocity = (
            cos_yaw * vx + sin_yaw * vy
        )

        lateral_velocity = (
            -sin_yaw * vx + cos_yaw * vy
        )

        total_speed = np.hypot(vx, vy)

        velocity = np.array(
            [
                forward_velocity / self.SPEED_SCALE,
                lateral_velocity / self.SPEED_SCALE,
                total_speed / self.SPEED_SCALE,
            ],
            dtype=np.float32,
        )

        self.last_velocity_readings = np.clip(
            velocity, -5.0, 5.0
        ).astype(np.float32)

        return self.last_velocity_readings.copy()

    # ----------------------------------------------------------
    # COMBINED OBSERVATION
    # ----------------------------------------------------------

    def _get_observation(self):
        base_observation = np.asarray(
            super()._get_observation(),
            dtype=np.float32,
        ).flatten()

        if base_observation.size != 19:
            raise RuntimeError(
                "Expected 19 original observations, but received "
                f"{base_observation.size}. Check CarEnv._get_observation()."
            )

        lidar = self.get_lidar_readings()
        imu = self.get_imu_readings()
        velocity = self.get_velocity_readings()

        observation = np.concatenate(
            [
                base_observation,
                lidar,
                imu,
                velocity,
            ]
        ).astype(np.float32)

        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        if observation.shape != (39,):
            raise RuntimeError(
                f"Expected observation shape (39,), got {observation.shape}."
            )

        return observation

    # ----------------------------------------------------------
    # STEERING + THROTTLE TO MOTOR COMMANDS
    # ----------------------------------------------------------

    def action_to_motor_commands(self, action):
        """
        Convert PPO action [steering, throttle] into differential
        left/right wheel commands.

        Positive steering turns right.
        Negative steering turns left.

        Both motors are kept forward-biased to avoid the policy
        accidentally commanding reverse during early training.
        """

        action = np.asarray(
            action, dtype=np.float32
        ).reshape(2)

        action = np.clip(action, -1.0, 1.0)

        steering = float(action[0])
        throttle_action = float(action[1])

        # Convert throttle [-1, 1] into [MIN_THROTTLE, MAX_THROTTLE].
        throttle = self.MIN_THROTTLE + (
            (throttle_action + 1.0) / 2.0
        ) * (
            self.MAX_THROTTLE - self.MIN_THROTTLE
        )

        steering_offset = (
            self.STEERING_STRENGTH * steering
        )

        left_motor = np.clip(
            throttle - steering_offset,
            0.05,
            1.0,
        )

        right_motor = np.clip(
            throttle + steering_offset,
            0.05,
            1.0,
        )

        motor_commands = np.array(
            [left_motor, right_motor],
            dtype=np.float32,
        )

        return action, steering, throttle, motor_commands

    # ----------------------------------------------------------
    # STEP
    # ----------------------------------------------------------

    def step(self, action):
        """
        Smooth policy actions, map them to motor commands, and use
        the original CarEnv.step() for physics, rewards, collision
        handling, and regeneration.
        """

        raw_action, steering, throttle, motor_commands = (
            self.action_to_motor_commands(action)
        )

        alpha = float(
            np.clip(self.ACTION_SMOOTHING, 0.0, 1.0)
        )

        smoothed_action = (
            alpha * raw_action
            + (1.0 - alpha) * self.previous_action_smoothed
        )

        smoothed_action = np.clip(
            smoothed_action, -1.0, 1.0
        ).astype(np.float32)

        self.previous_action_smoothed = smoothed_action.copy()

        # Recompute motor commands after smoothing.
        _, steering, throttle, motor_commands = (
            self.action_to_motor_commands(smoothed_action)
        )

        # Bypass GhostCarEnv.step so the mapped motor commands go
        # directly to the original CarEnv.step implementation.
        result = super(GhostCarEnv, self).step(motor_commands)

        observation, reward, terminated, truncated, info = result

        # Update the acceleration baseline after physics and
        # observation calculation. Also handles reset/regeneration.
        self.previous_sensor_velocity = self.data.qvel[0:3].copy()

        info["ghost_id"] = self.ghost_id
        info["raw_action"] = raw_action.copy()
        info["smoothed_action"] = smoothed_action.copy()
        info["steering_command"] = float(steering)
        info["throttle_command"] = float(throttle)
        info["motor_commands"] = motor_commands.copy()

        info["lidar"] = self.last_lidar_readings.copy()
        info["imu"] = self.last_imu_readings.copy()
        info["velocity"] = self.last_velocity_readings.copy()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )
