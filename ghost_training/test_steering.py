
import numpy as np
from ghost_training.sensor_car_env import SensorGhostCarEnv


def run_test(name, steering, throttle):
    env = SensorGhostCarEnv(ghost_id=0)

    try:
        obs, info = env.reset(seed=42)
        start_pos = env.data.qpos[:2].copy()
        start_yaw = env.get_yaw()

        action = np.array(
            [steering, throttle],
            dtype=np.float32
        )

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        end_pos = env.data.qpos[:2].copy()
        end_yaw = env.get_yaw()

        yaw_change = np.arctan2(
            np.sin(end_yaw - start_yaw),
            np.cos(end_yaw - start_yaw)
        )

        print(f"\n{name}")
        print("Start position:", np.round(start_pos, 3))
        print("End position:  ", np.round(end_pos, 3))
        print("Yaw change:    ", round(float(yaw_change), 3))
        print("Motor commands:", np.round(info["motor_commands"], 3))
        print("Regenerated:   ", info.get("regenerated", False))

    finally:
        env.close()


if __name__ == "__main__":
    run_test("STRAIGHT", 0.0, 1.0)
    run_test("LEFT", -0.7, 1.0)
    run_test("RIGHT", 0.7, 1.0)
