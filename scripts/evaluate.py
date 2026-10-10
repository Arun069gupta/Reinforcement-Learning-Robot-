import time

import mujoco
import mujoco.viewer

from stable_baselines3 import PPO

from environments.car_env import CarEnv


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

env = CarEnv()


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = PPO.load(
    "training/ppo_car",
    env=env
)


print("\n==============================")
print("RL CAR EVALUATION")
print("==============================")

print("Trained model loaded.")


# ==========================================
# RESET ENVIRONMENT
# ==========================================

obs, info = env.reset()

print("Environment reset.")


# ==========================================
# MUJOCO VIEWER
# ==========================================

with mujoco.viewer.launch_passive(
    env.model,
    env.data
) as viewer:

    step = 0

    while viewer.is_running():

        # ----------------------------------
        # PPO PREDICTION
        # ----------------------------------

        action, _ = model.predict(
            obs,
            deterministic=True
        )


        # ----------------------------------
        # ENVIRONMENT STEP
        # ----------------------------------

        obs, reward, terminated, truncated, info = env.step(
            action
        )


        # ----------------------------------
        # PRINT STATUS
        # ----------------------------------

        if step % 100 == 0:

            print(
                f"Step: {step:5d} | "
                f"Progress: {info['progress']:.3f} | "
                f"Distance: {info['distance_from_centerline']:.3f} | "
                f"Velocity: {info['forward_velocity']:.3f} | "
                f"Reward: {reward:.3f}"
            )


        # ----------------------------------
        # UPDATE VIEWER
        # ----------------------------------

        viewer.sync()

        step += 1

        time.sleep(0.002)


        # ----------------------------------
        # EPISODE END
        # ----------------------------------

        if terminated or truncated:

            print("\n==============================")

            print("EPISODE FINISHED")

            print(
                f"Final progress: "
                f"{info['progress']:.3f}"
            )

            print(
                f"Lap count: "
                f"{info['lap']}"
            )

            print("==============================\n")


            # Reset for another episode

            obs, info = env.reset()

            step = 0

