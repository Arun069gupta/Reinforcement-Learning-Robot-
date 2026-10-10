
import numpy as np

from ghost_training.sensor_car_env import SensorGhostCarEnv


def main():
    env = SensorGhostCarEnv(ghost_id=0)

    try:
        observation, info = env.reset(seed=42)

        print("\n========== SENSOR TEST ==========")
        print("Observation shape:", observation.shape)
        print("Expected shape:   ", (39,))
        print("All values finite:", np.isfinite(observation).all())

        print("\nLiDAR readings:")
        print(env.get_lidar_readings())

        print("\nIMU readings:")
        print(env.get_imu_readings())

        print("\nVelocity readings:")
        print(env.get_velocity_readings())

        assert observation.shape == (39,)
        assert np.isfinite(observation).all()

        print("\nTesting one action...")
        observation, reward, terminated, truncated, info = (
            env.step(np.array([0.5, 0.5], dtype=np.float32))
        )

        print("New observation shape:", observation.shape)
        print("Reward:", reward)
        print("Sensor values finite:", np.isfinite(observation).all())

        assert observation.shape == (39,)
        assert np.isfinite(observation).all()

        print("\nSensor test passed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
