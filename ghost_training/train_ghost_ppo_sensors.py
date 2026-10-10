
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from ghost_training.sensor_car_env import SensorGhostCarEnv


# ==========================================================
# TRAINING SETTINGS
# ==========================================================

NUM_GHOST_CARS = 4

# Additional timesteps, continuing from the saved checkpoint.
TOTAL_TIMESTEPS = 100_000

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "ghost_training",
    "ppo_ghost_car_sensors_no_viewer",
)


# ==========================================================
# ENVIRONMENT FACTORY
# ==========================================================

def make_ghost_env(ghost_id):
    def create():
        return SensorGhostCarEnv(ghost_id=ghost_id)

    return create


# ==========================================================
# TRAINING DIAGNOSTICS
# ==========================================================

class SensorTrainingCallback(BaseCallback):
    def __init__(self, log_interval=500, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.next_log = log_interval

    def _on_step(self):
        if self.num_timesteps >= self.next_log:
            try:
                progress_values = self.training_env.get_attr(
                    "current_progress"
                )
                best_values = self.training_env.get_attr(
                    "best_progress"
                )
                attempts = self.training_env.get_attr(
                    "attempt_number"
                )
                positions = self.training_env.get_attr("data")

                print(
                    f"\n[PPO] Timesteps: {self.num_timesteps:,}"
                )
                print(
                    "Progress:",
                    np.round(progress_values, 3).tolist(),
                )
                print(
                    "Best progress:",
                    np.round(best_values, 3).tolist(),
                )
                print("Attempts:", attempts)

                for i, data in enumerate(positions):
                    position = data.qpos[0:2]
                    speed = float(
                        np.linalg.norm(data.qvel[0:2])
                    )

                    print(
                        f"Car {i}: "
                        f"position={np.round(position, 2).tolist()}, "
                        f"speed={speed:.3f} m/s"
                    )

            except Exception as exc:
                print(f"[PPO] Diagnostic logging failed: {exc}")

            while self.next_log <= self.num_timesteps:
                self.next_log += self.log_interval

        return True


# ==========================================================
# MAIN TRAINING
# ==========================================================

def main():
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Validate the sensor environment.
    # ------------------------------------------------------

    print("\nChecking sensor environment...")

    test_env = SensorGhostCarEnv(ghost_id=0)

    try:
        check_env(test_env, warn=True)

        observation, _ = test_env.reset(seed=42)

        print("Observation shape:", observation.shape)
        print("Expected shape:   ", (39,))
        print(
            "Finite observations:",
            bool(np.isfinite(observation).all()),
        )

        assert observation.shape == (39,)
        assert np.isfinite(observation).all()

    finally:
        test_env.close()

    print("Environment check passed.")

    # ------------------------------------------------------
    # Create four environments sharing one PPO policy.
    # ------------------------------------------------------

    envs = DummyVecEnv(
        [
            make_ghost_env(i)
            for i in range(NUM_GHOST_CARS)
        ]
    )

    try:
        # --------------------------------------------------
        # Load the existing checkpoint.
        # --------------------------------------------------

        if not os.path.isfile(MODEL_PATH + ".zip"):
            raise FileNotFoundError(
                "PPO checkpoint not found:\n"
                f"{MODEL_PATH}.zip\n"
                "Run the initial training script first."
            )

        print("\nLoading saved PPO checkpoint...")

        model = PPO.load(
            MODEL_PATH,
            env=envs,
            device="auto",
        )

        print("Checkpoint loaded successfully.")

        print("\n======================================")
        print("CONTINUING FOUR-CAR SENSOR PPO")
        print("======================================")
        print(f"Environments          : {NUM_GHOST_CARS}")
        print(f"Additional timesteps  : {TOTAL_TIMESTEPS:,}")
        print("Observation size      : 39")
        print("Action size           : 2")
        print("Shared PPO policy     : ENABLED")
        print("Live viewer           : DISABLED")
        print(f"Checkpoint            : {MODEL_PATH}.zip")
        print("======================================\n")

        callback = SensorTrainingCallback(
            log_interval=500
        )

        # reset_num_timesteps=False preserves the global
        # timestep counter when continuing training.
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        # Save the updated policy to the same checkpoint.
        model.save(MODEL_PATH)

        print(
            f"\nUpdated model saved to:\n{MODEL_PATH}.zip"
        )

    finally:
        envs.close()

    print("\nContinued training run finished.")


if __name__ == "__main__":
    main()
