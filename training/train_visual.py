import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from environments.car_env import CarEnv


env = CarEnv()

# Allow the visual test to run longer than the training episode limit.
# This does NOT change the trained PPO model.
if hasattr(env, "max_steps"):
    env.max_steps = 20000

model = PPO.load(
    "training/ppo_car",
    env=env
)

obs, info = env.reset()


with mujoco.viewer.launch_passive(
    env.model,
    env.data
) as viewer:

    print("\n==========================================")
    print("VISUAL TEST STARTED")
    print("==========================================")
    print("Maximum visual steps : 20000")
    print("Press Ctrl+C in terminal to stop.")
    print("==========================================\n")

    step = 0

    while viewer.is_running():

        action, _states = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        step += 1

        viewer.sync()

        if step % 50 == 0:

            position = env.data.qpos[0:2]

            print(
                f"Step {step:5d} | "
                f"Action [{action[0]: .3f}, {action[1]: .3f}] | "
                f"Pos [{position[0]: .3f}, {position[1]: .3f}] | "
                f"Progress {info.get('progress', 0.0):.3f} | "
                f"Segment {info.get('segment', 0):3d} | "
                f"Distance {info.get('distance', 0.0):.3f} | "
                f"Forward {info.get('forward_velocity', 0.0):.3f} | "
                f"Safe {info.get('safe_line', True)} | "
                f"Recovery {info.get('recovery_distance', 0.0):.3f} | "
                f"Attempt {info.get('attempt_number', 0)}"
            )

        if terminated or truncated:

            print("\n==========================================")
            print("EPISODE FINISHED")
            print("==========================================")

            print(
                f"Final Progress : "
                f"{info.get('progress', 0.0):.3f}"
            )

            print(
                f"Final Segment  : "
                f"{info.get('segment', 0)}"
            )

            print(
                f"Final Distance : "
                f"{info.get('distance', 0.0):.3f}"
            )

            print(
                f"Final Reward   : "
                f"{reward:.3f}"
            )

            print(
                f"Attempt        : "
                f"{info.get('attempt_number', 0)}"
            )

            print("==========================================\n")

            obs, info = env.reset()

        time.sleep(0.002)


env.close()