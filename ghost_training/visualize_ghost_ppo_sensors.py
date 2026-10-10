
import os
import copy
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from ghost_training.sensor_car_env import SensorGhostCarEnv


# ==========================================================
# SETTINGS
# ==========================================================

NUM_GHOST_CARS = 4

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CAR_XML = os.path.join(
    PROJECT_DIR,
    "models",
    "car.xml",
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "ghost_training",
    "ppo_ghost_car_sensors_no_viewer.zip",
)

COLORS = [
    "0.1 0.3 1.0 1",   # Blue
    "0.1 0.9 0.3 1",   # Green
    "1.0 0.4 0.1 1",   # Orange
    "0.8 0.2 1.0 1",   # Purple
]

# DEBUGGING ONLY:
# These offsets separate the cars so you can confirm that all
# four bodies render. They are not guaranteed to be valid
# positions on the racing track.
VISUAL_OFFSETS = [
    (0.0, 0.0),
    (0.0, 1.5),
    (0.0, 3.0),
    (0.0, 4.5),
]

LOG_INTERVAL_SECONDS = 2.0


# ==========================================================
# BUILD FOUR-CAR DISPLAY MODEL
# ==========================================================

def build_visual_model():
    tree = ET.parse(CAR_XML)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    asset = root.find("asset")
    actuator = root.find("actuator")

    if worldbody is None or actuator is None:
        raise RuntimeError(
            "car.xml must contain worldbody and actuator."
        )

    if asset is None:
        asset = ET.Element("asset")
        root.insert(list(root).index(worldbody), asset)

    original_car = worldbody.find(
        "./body[@name='car']"
    )

    if original_car is None:
        raise RuntimeError(
            "Could not find body named 'car' in models/car.xml."
        )

    car_template = copy.deepcopy(original_car)
    worldbody.remove(original_car)

    original_actuators = [
        copy.deepcopy(item)
        for item in list(actuator)
    ]

    actuator.clear()

    for i in range(NUM_GHOST_CARS):
        car = copy.deepcopy(car_template)
        renamed = {}

        # Rename all named elements so every car has its own
        # joints, geoms, sites and other named objects.
        for element in car.iter():
            old_name = element.get("name")

            if old_name:
                new_name = f"{old_name}_ghost{i}"
                renamed[old_name] = new_name
                element.set("name", new_name)

            elif element.tag == "freejoint":
                element.set(
                    "name",
                    f"car_freejoint_ghost{i}",
                )

            if element.tag == "geom":
                # Rendering only. Physics runs in each real env.
                element.set("contype", "0")
                element.set("conaffinity", "0")

                if old_name == "chassis":
                    element.set(
                        "material",
                        f"ghost_material_{i}",
                    )

        ET.SubElement(
            asset,
            "material",
            name=f"ghost_material_{i}",
            rgba=COLORS[i],
        )

        worldbody.append(car)

        # Copy actuators and point them at this car's joints.
        for original in original_actuators:
            copied = copy.deepcopy(original)

            actuator_name = copied.get("name")
            if actuator_name:
                copied.set(
                    "name",
                    f"{actuator_name}_ghost{i}",
                )

            joint_name = copied.get("joint")

            if joint_name:
                if joint_name not in renamed:
                    raise RuntimeError(
                        "Actuator references unknown joint: "
                        f"{joint_name}"
                    )

                copied.set(
                    "joint",
                    renamed[joint_name],
                )

            actuator.append(copied)

    visual_model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode")
    )

    # Verify all four display free joints exist.
    print("\nDisplay model verification:")

    for i in range(NUM_GHOST_CARS):
        name = f"car_freejoint_ghost{i}"

        joint_id = mujoco.mj_name2id(
            visual_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        print(f"  {name}: joint_id={joint_id}")

        if joint_id < 0:
            raise RuntimeError(
                f"Missing display joint: {name}"
            )

    return visual_model


# ==========================================================
# SYNC TRAINING CARS TO DISPLAY MODEL
# ==========================================================

def sync_ghosts(envs, visual_model, visual_data):
    for i, env in enumerate(envs):
        # Copy the car's free-joint position and orientation.
        joint_name = f"car_freejoint_ghost{i}"

        target_joint_id = mujoco.mj_name2id(
            visual_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if target_joint_id < 0:
            raise RuntimeError(
                f"Missing display joint: {joint_name}"
            )

        target_qpos = int(
            visual_model.jnt_qposadr[target_joint_id]
        )

        visual_data.qpos[
            target_qpos:target_qpos + 7
        ] = env.data.qpos[0:7]

        # Copy rear wheel rotation for visual wheel motion.
        for wheel_name in (
            "rear_left_joint",
            "rear_right_joint",
        ):
            source_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                wheel_name,
            )

            target_id = mujoco.mj_name2id(
                visual_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{wheel_name}_ghost{i}",
            )

            if source_id >= 0 and target_id >= 0:
                source_qpos = int(
                    env.model.jnt_qposadr[source_id]
                )

                target_wheel_qpos = int(
                    visual_model.jnt_qposadr[target_id]
                )

                visual_data.qpos[target_wheel_qpos] = (
                    env.data.qpos[source_qpos]
                )

    mujoco.mj_forward(visual_model, visual_data)


# ==========================================================
# PRINT CURRENT STATE
# ==========================================================

def print_status(envs, step_count):
    print(f"\nDisplay step: {step_count}")

    for i, env in enumerate(envs):
        position = env.data.qpos[0:3]
        speed = float(
            np.linalg.norm(env.data.qvel[0:2])
        )

        progress = float(
            getattr(env, "current_progress", 0.0)
        )

        best_progress = float(
            getattr(env, "best_progress", 0.0)
        )

        print(
            f"Car {i} | "
            f"Position: {np.round(position, 2)} | "
            f"Speed: {speed:.2f} m/s | "
            f"Progress: {progress:.3f} | "
            f"Best: {best_progress:.3f}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():
    if not os.path.isfile(CAR_XML):
        raise FileNotFoundError(
            f"Car XML not found:\n{CAR_XML}"
        )

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"PPO model not found:\n{MODEL_PATH}"
        )

    print("Loading saved PPO model...")
    model = PPO.load(
        MODEL_PATH,
        device="cpu",
    )

    envs = [
        SensorGhostCarEnv(ghost_id=i)
        for i in range(NUM_GHOST_CARS)
    ]

    try:
        # --------------------------------------------------
        # Reset and separate the cars for visual debugging.
        # --------------------------------------------------

        print("\nResetting four environments...")

        for i, env in enumerate(envs):
            env.reset(seed=100 + i)

            dx, dy = VISUAL_OFFSETS[i]

            env.data.qpos[0] += dx
            env.data.qpos[1] += dy

            mujoco.mj_forward(
                env.model,
                env.data,
            )

            # Refresh sensor baseline after changing position.
            env.previous_sensor_velocity = (
                env.data.qvel[0:3].copy()
            )

            print(
                f"Car {i} initial position:",
                np.round(env.data.qpos[0:3], 3),
            )

        # --------------------------------------------------
        # Create display model.
        # --------------------------------------------------

        visual_model = build_visual_model()
        visual_data = mujoco.MjData(visual_model)

        sync_ghosts(
            envs,
            visual_model,
            visual_data,
        )

        print("\n==========================================")
        print("FOUR-CAR SENSOR PPO VISUALIZATION")
        print("==========================================")
        print("Car 0: Blue")
        print("Car 1: Green")
        print("Car 2: Orange")
        print("Car 3: Purple")
        print("Policy: Saved PPO checkpoint")
        print("Training: Disabled")
        print("Close the viewer to stop.")
        print("==========================================")

        # --------------------------------------------------
        # Launch the viewer.
        # --------------------------------------------------

        with mujoco.viewer.launch_passive(
            visual_model,
            visual_data,
        ) as viewer:

            viewer.cam.lookat[:] = [
                0.0,
                0.0,
                0.0,
            ]

            viewer.cam.distance = 22.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -90

            viewer.sync()

            step_count = 0
            last_log_time = time.time()

            while viewer.is_running():

                # Each environment advances independently.
                # All four use the same saved PPO policy.
                for env in envs:
                    observation = env._get_observation()

                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )

                    _, _, terminated, truncated, _ = (
                        env.step(action)
                    )

                    if terminated or truncated:
                        env.reset()

                # Synchronize all four cars into the viewer.
                sync_ghosts(
                    envs,
                    visual_model,
                    visual_data,
                )

                viewer.sync()
                step_count += 1

                now = time.time()

                if (
                    now - last_log_time
                    >= LOG_INTERVAL_SECONDS
                ):
                    print_status(
                        envs,
                        step_count,
                    )

                    last_log_time = now

    finally:
        for env in envs:
            env.close()

    print("\nVisualization closed.")


if __name__ == "__main__":
    main()
