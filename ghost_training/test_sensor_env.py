
import numpy as np

from ghost_training.sensor_car_env import SensorGhostCarEnv


def main():
    env = SensorGhostCarEnv(ghost_id=0)

    try:
        obs, info = env.reset(seed=42)

        print("\n========== ENVIRONMENT TEST ==========")
        print("Observation shape:", obs.shape)
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Initial position:", env.data.qpos[:3])

        assert obs.shape == (39,), (
            f"Expected (39,), received {obs.shape}"
        )
        assert np.all(np.isfinite(obs)), (
            "Observation contains NaN or infinity"
        )

        # Test a forward-driving action:
        # steering = 0, throttle = 1.
        action = np.array([0.0, 1.0], dtype=np.float32)

        print("\nTesting forward action:", action)

        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"Step {step + 1:02d} | "
                f"Reward: {reward:8.3f} | "
                f"Position: {np.round(env.data.qpos[:3], 3)} | "
                f"Velocity: {np.round(info['velocity'], 3)} | "
                f"Motors: {np.round(info['motor_commands'], 3)}"
            )

            assert obs.shape == (39,)
            assert np.all(np.isfinite(obs))

            if terminated or truncated:
                print("Episode ended or environment reset.")
                break

        print("\nLiDAR:", np.round(info["lidar"], 3))
        print("IMU:", np.round(info["imu"], 3))
        print("Velocity:", np.round(info["velocity"], 3))
        print("\nEnvironment test completed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
