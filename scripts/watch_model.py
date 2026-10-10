import time
import os

from stable_baselines3 import PPO

from environments.car_env import CarEnv


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "training/ppo_car.zip"


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

env = CarEnv()


# ==========================================
# LOAD TRAINED PPO MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found:")
    print(MODEL_PATH)
    print()
    print("Train the model first.")
    exit()

print("Loading trained PPO model...")

model = PPO.load(
    MODEL_PATH,
    env=env
)

print("Model loaded successfully.")


# ==========================================
# RESET ENVIRONMENT
# ==========================================

obs, info = env.reset()


# ==========================================
# OPEN MUJOCO VIEWER
# ==========================================

import mujoco.viewer

print()
print("====================================")
print(" PPO CAR VISUALIZATION")
print("====================================")
print("Close the MuJoCo window to stop.")
print()

with mujoco.viewer.launch_passive(
    env.model,
    env.data
) as viewer:

    viewer.cam.distance = 8.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -25

    while viewer.is_running():

        # ==================================
        # PPO PREDICTION
        # ==================================

        action, _states = model.predict(
            obs,
            deterministic=True
        )

        # ==================================
        # ENVIRONMENT STEP
        # ==================================

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        # ==================================
        # PRINT STATUS
        # ==================================

        if env.step_count % 100 == 0:

            print(
                f"Step: {env.step_count:5d} | "
                f"Progress: {info['progress']:.3f} | "
                f"Distance: {info['distance_from_centerline']:.3f} | "
                f"Velocity: {info['forward_velocity']:.3f} | "
                f"Reward: {reward:.3f}"
            )

        # ==================================
        # UPDATE VIEWER
        # ==================================

        viewer.sync()

        # ==================================
        # CONTROL SIMULATION SPEED
        # ==================================

        time.sleep(0.002)

        # ==================================
        # EPISODE FINISHED
        # ==================================

        if terminated or truncated:

            print()
            print("====================================")
            print("EPISODE FINISHED")
            print(
                f"Final Progress: {info['progress']:.3f}"
            )
            print(
                f"Lap Count: {info['lap']}"
            )
            print("====================================")
            print()

            obs, info = env.reset()
