import os
import time

import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from environments.car_env import CarEnv


# ==============================================================
# LIVE MUJOCO VIEWER CALLBACK
# ==============================================================

class LiveViewerCallback(BaseCallback):

    def __init__(self, env, viewer, verbose=0):

        super().__init__(verbose)

        self.env = env
        self.viewer = viewer

    def _on_step(self) -> bool:

        # ------------------------------------------------------
        # Update MuJoCo viewer
        # ------------------------------------------------------

        if self.viewer.is_running():

            self.viewer.sync()

        else:

            # If the user closes the viewer,
            # stop PPO training.
            return False

        # ------------------------------------------------------
        # Print useful information periodically
        # ------------------------------------------------------

        if self.num_timesteps % 1000 == 0:

            try:

                info = self.locals["infos"][0]

                progress = info.get(
                    "progress",
                    0.0
                )

                best_progress = info.get(
                    "best_progress",
                    0.0
                )

                forward_velocity = info.get(
                    "forward_velocity",
                    0.0
                )

                distance = info.get(
                    "distance",
                    0.0
                )

                attempt = info.get(
                    "attempt",
                    0
                )

                failed_routes = info.get(
                    "failed_routes",
                    0
                )

                print(
                    f"\n"
                    f"[LIVE TRAINING] "
                    f"Step: {self.num_timesteps:7d} | "
                    f"Progress: {progress:.3f} | "
                    f"Best: {best_progress:.3f} | "
                    f"Forward: {forward_velocity:.3f} | "
                    f"Distance: {distance:.3f} | "
                    f"Attempt: {attempt} | "
                    f"Failures: {failed_routes}"
                )

            except Exception:
                pass

        # ------------------------------------------------------
        # Small delay so viewer can update smoothly
        # ------------------------------------------------------

        time.sleep(0.001)

        return True


# ==============================================================
# CREATE ENVIRONMENT
# ==============================================================

env = CarEnv()


# ==============================================================
# CHECK ENVIRONMENT
# ==============================================================

print("\n==========================================")
print("CHECKING ENVIRONMENT")
print("==========================================\n")

check_env(
    env,
    warn=True
)

print("\nEnvironment check completed.\n")


# ==============================================================
# MODEL PATH
# ==============================================================

model_path = "training/ppo_car"


# ==============================================================
# REMOVE OLD MODEL
# ==============================================================
#
# We are training a fresh model because the reward/environment
# has changed.
# ==============================================================

old_files = [
    model_path + ".zip"
]

for file_path in old_files:

    if os.path.exists(file_path):

        print(
            f"Removing old model: {file_path}"
        )

        os.remove(file_path)


# ==============================================================
# CREATE PPO MODEL
# ==============================================================

model = PPO(

    policy="MlpPolicy",

    env=env,

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

    device="auto"
)


# ==============================================================
# START MUJOCO LIVE VIEWER
# ==============================================================

print("\n==========================================")
print("STARTING LIVE MUJOCO VIEWER")
print("==========================================")
print("The car will now be visible while PPO trains.")
print("")
print("IMPORTANT:")
print("- Do NOT close the MuJoCo window.")
print("- Closing the window stops training.")
print("- The car behavior will change as PPO learns.")
print("==========================================\n")


with mujoco.viewer.launch_passive(
    env.model,
    env.data
) as viewer:

    # ==========================================================
    # INITIAL VIEW
    # ==========================================================

    viewer.sync()


    # ==========================================================
    # CALLBACK
    # ==========================================================

    callback = LiveViewerCallback(
        env=env,
        viewer=viewer
    )


    # ==========================================================
    # TRAINING
    # ==========================================================

    print("\n==========================================")
    print("STARTING PPO TRAINING")
    print("==========================================")
    print("Timesteps          : 500000")
    print("Observation size   : 19")
    print("Action size        : 2")
    print("Track              : Enlarged + Rounded")
    print("Safe corridor      : 0.50 m")
    print("Recovery system    : ENABLED")
    print("Live visualization : ENABLED")
    print("==========================================\n")


    model.learn(

        total_timesteps=500_000,

        callback=callback
    )


    # ==========================================================
    # SAVE MODEL
    # ==========================================================

    model.save(
        model_path
    )


    # ==========================================================
    # FINAL VIEW UPDATE
    # ==========================================================

    viewer.sync()


# ==============================================================
# COMPLETE
# ==============================================================

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(
    f"Model saved to: "
    f"{model_path}.zip"
)

print("==========================================\n")


# ==============================================================
# CLOSE ENVIRONMENT
# ==============================================================

env.close()