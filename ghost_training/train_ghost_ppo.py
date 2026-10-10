
import os
import time
import copy
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from ghost_training.ghost_car_env import SensorGhostCarEnv


# ==============================================================
# SETTINGS
# ==============================================================

NUM_GHOST_CARS = 4
TOTAL_TIMESTEPS = 500_000

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAR_XML = os.path.join(PROJECT_DIR, "models", "car.xml")
MODEL_PATH = os.path.join(
    PROJECT_DIR, "ghost_training", "ppo_ghost_car"
)

COLORS = [
    "0.1 0.3 1.0 1",   # Blue
    "0.1 0.9 0.3 1",   # Green
    "1.0 0.4 0.1 1",   # Orange
    "0.8 0.2 1.0 1",   # Purple
]


# ==============================================================
# BUILD ONE VISUAL MODEL WITH FOUR CARS
# ==============================================================

def build_visual_model():

    tree = ET.parse(CAR_XML)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    asset = root.find("asset")
    actuator = root.find("actuator")

    if worldbody is None or actuator is None:
        raise RuntimeError(
            "car.xml must contain worldbody and actuator elements."
        )

    original_car = worldbody.find("./body[@name='car']")

    if original_car is None:
        raise RuntimeError(
            "Could not find body named 'car' in models/car.xml."
        )

    # Remove the original car from the visualization scene.
    worldbody.remove(original_car)

    # Save the original actuators before replacing them.
    original_actuators = list(actuator)
    actuator.clear()

    # Create four independently named car copies.
    for i in range(NUM_GHOST_CARS):

        car = copy.deepcopy(original_car)
        joint_names = {}

        # Rename every named element to prevent duplicate names.
        for element in car.iter():

            old_name = element.get("name")

            if old_name:
                new_name = f"{old_name}_ghost{i}"
                joint_names[old_name] = new_name
                element.set("name", new_name)

            elif element.tag == "freejoint":
                element.set("name", f"car_freejoint_ghost{i}")

            # This model is for display only. It does not calculate
            # car-to-car contact physics.
            if element.tag == "geom":
                element.set("contype", "0")
                element.set("conaffinity", "0")

                if old_name == "chassis":
                    element.set("material", f"ghost_material_{i}")

        # Give each chassis a distinctive color.
        material = ET.SubElement(
            asset,
            "material",
            name=f"ghost_material_{i}",
            rgba=COLORS[i],
        )

        worldbody.append(car)

        # Duplicate actuators and point them at the copied joints.
        for original_actuator in original_actuators:

            copied_actuator = copy.deepcopy(original_actuator)

            old_actuator_name = copied_actuator.get("name")

            if old_actuator_name:
                copied_actuator.set(
                    "name",
                    f"{old_actuator_name}_ghost{i}"
                )

            joint_name = copied_actuator.get("joint")

            if joint_name:
                copied_actuator.set(
                    "joint",
                    joint_names[joint_name]
                )

            actuator.append(copied_actuator)

    xml_string = ET.tostring(
        root,
        encoding="unicode"
    )

    return mujoco.MjModel.from_xml_string(xml_string)


# ==============================================================
# SYNCHRONIZE THE DISPLAY WITH THE TRAINING CARS
# ==============================================================

def sync_ghosts(training_envs, visual_model, visual_data):

    for i, env in enumerate(training_envs):

        # Locate this ghost's free joint in the display model.
        free_joint_id = mujoco.mj_name2id(
            visual_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"car_freejoint_ghost{i}"
        )

        if free_joint_id < 0:
            raise RuntimeError(
                f"Could not find free joint for ghost {i}."
            )

        visual_qpos = int(
            visual_model.jnt_qposadr[free_joint_id]
        )

        # The original CarEnv stores the car's free-joint pose
        # in its first seven generalized position values.
        visual_data.qpos[
            visual_qpos:visual_qpos + 7
        ] = env.data.qpos[0:7]

        # Synchronize both driven-wheel angles.
        for wheel_name in (
            "rear_left_joint",
            "rear_right_joint",
        ):

            source_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                wheel_name
            )

            display_id = mujoco.mj_name2id(
                visual_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{wheel_name}_ghost{i}"
            )

            if source_id >= 0 and display_id >= 0:

                source_qpos = int(
                    env.model.jnt_qposadr[source_id]
                )

                display_qpos = int(
                    visual_model.jnt_qposadr[display_id]
                )

                visual_data.qpos[display_qpos] = (
                    env.data.qpos[source_qpos]
                )

    # Recalculate displayed body positions from the synchronized pose.
    mujoco.mj_forward(visual_model, visual_data)


# ==============================================================
# LIVE VIEWER CALLBACK
# ==============================================================

class GhostViewerCallback(BaseCallback):

    def __init__(
        self,
        viewer,
        visual_model,
        visual_data,
        verbose=0,
    ):

        super().__init__(verbose)

        self.viewer = viewer
        self.visual_model = visual_model
        self.visual_data = visual_data

    def _on_step(self):

        if not self.viewer.is_running():
            return False

        # DummyVecEnv exposes its four independent environments.
        training_envs = self.training_env.envs

        sync_ghosts(
            training_envs,
            self.visual_model,
            self.visual_data
        )

        self.viewer.sync()

        if self.num_timesteps % 1000 < NUM_GHOST_CARS:

            infos = self.locals.get("infos", [])

            if infos:

                progress = [
                    round(info.get("progress", 0.0), 3)
                    for info in infos
                ]

                print(
                    f"[GHOST PPO] "
                    f"Steps: {self.num_timesteps:,} | "
                    f"Progress: {progress}"
                )

        return True


# ==============================================================
# CREATE GHOST ENVIRONMENTS
# ==============================================================

def make_ghost_env(ghost_id):

    def create():
        return SensorGhostCarEnv(ghost_id=ghost_id)

    return create


def main():

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    print("\nChecking the sensor-enabled environment...")

    test_env = SensorGhostCarEnv(ghost_id=0)

    try:
        check_env(test_env, warn=True)
    finally:
        test_env.close()

    print("Sensor environment check passed.")

    envs = DummyVecEnv(
        [
            make_ghost_env(i)
            for i in range(NUM_GHOST_CARS)
        ]
    )

    try:

        print("Building the combined visualization model...")

        visual_model = build_visual_model()
        visual_data = mujoco.MjData(visual_model)

        # Reset the four training environments and synchronize
        # their starting poses into the display.
        envs.reset()

        sync_ghosts(
            envs.envs,
            visual_model,
            visual_data
        )

        model = PPO(
            policy="MlpPolicy",
            env=envs,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto",
        )

        print("\n==========================================")
        print("STARTING FOUR-GHOST PPO TRAINING")
        print("==========================================")
        print(f"Ghost cars        : {NUM_GHOST_CARS}")
        print(f"Timesteps         : {TOTAL_TIMESTEPS:,}")
        print("Shared PPO policy : ENABLED")
        print("Live viewer       : ENABLED")
        print("==========================================\n")

        with mujoco.viewer.launch_passive(
            visual_model,
            visual_data
        ) as viewer:

            # Set a useful overhead view of the track.
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            viewer.cam.distance = 18.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -90

            viewer.sync()

            callback = GhostViewerCallback(
                viewer=viewer,
                visual_model=visual_model,
                visual_data=visual_data,
            )

            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callback,
            )

            # Save even if training stopped because the viewer closed.
            model.save(MODEL_PATH)

            print(f"\nModel saved to: {MODEL_PATH}.zip")

    finally:
        envs.close()


if __name__ == "__main__":
    main()
