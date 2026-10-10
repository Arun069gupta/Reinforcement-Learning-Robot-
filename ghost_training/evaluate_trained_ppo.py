
import os
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from ghost_training.sensor_car_env import SensorGhostCarEnv


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "ghost_training",
    "ppo_ghost_car_sensors_no_viewer.zip",
)

NUM_EPISODES = 3
MAX_STEPS_PER_EPISODE = 2000
SHOW_VIEWER = True


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("Loading saved PPO policy...")
    model = PPO.load(MODEL_PATH, device="cpu")

    env = SensorGhostCarEnv(ghost_id=0)

    try:
        obs, info = env.reset(seed=123)

        print("\n========== TRAINED POLICY ==========")
        print("Model:", MODEL_PATH)
        print("Observation shape:", obs.shape)
        print("Action space:", env.action_space)
        print("Episodes:", NUM_EPISODES)
        print("====================================")

        viewer_context = (
            mujoco.viewer.launch_passive(env.model, env.data)
            if SHOW_VIEWER
            else None
        )

        def run_episode(viewer=None):
            obs, info = env.reset()
            start_pos = env.data.qpos[:3].copy()

            total_reward = 0.0
            max_progress = 0.0
            collisions = 0
            completed_steps = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                completed_steps += 1

                progress = float(
                    getattr(env, "current_progress", 0.0)
                )
                max_progress = max(max_progress, progress)

                if info.get("wall_collision", False):
                    collisions += 1

                if viewer is not None:
                    if not viewer.is_running():
                        print("Viewer closed.")
                        return None

                    viewer.sync()

                if step % 100 == 0:
                    position = np.round(
                        env.data.qpos[:2], 2
                    )

                    print(
                        f"Step {step:4d} | "
                        f"Position {position} | "
                        f"Progress {progress:.3f} | "
                        f"Reward {reward:.3f}"
                    )

                if terminated or truncated:
                    break

            final_pos = env.data.qpos[:3].copy()

            return {
                "steps": completed_steps,
                "reward": total_reward,
                "max_progress": max_progress,
                "collisions": collisions,
                "start_position": start_pos,
                "final_position": final_pos,
                "best_progress": float(
                    getattr(env, "best_progress", 0.0)
                ),
            }

        if viewer_context is not None:
            with viewer_context as viewer:
                viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
                viewer.cam.distance = 18.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -90

                results = []

                for episode in range(NUM_EPISODES):
                    print(f"\n========== EPISODE {episode + 1} ==========")
                    result = run_episode(viewer)

                    if result is None:
                        break

                    results.append(result)
                    print("Episode result:", result)

                print("\n========== EVALUATION SUMMARY ==========")
                for i, result in enumerate(results, start=1):
                    print(
                        f"Episode {i}: "
                        f"steps={result['steps']}, "
                        f"reward={result['reward']:.2f}, "
                        f"max_progress={result['max_progress']:.3f}, "
                        f"collisions={result['collisions']}, "
                        f"best_progress={result['best_progress']:.3f}"
                    )
        else:
            results = []

            for episode in range(NUM_EPISODES):
                print(f"\n========== EPISODE {episode + 1} ==========")
                result = run_episode()

                if result is not None:
                    results.append(result)

            print("\nEvaluation results:", results)

    finally:
        env.close()


if __name__ == "__main__":
    main()
