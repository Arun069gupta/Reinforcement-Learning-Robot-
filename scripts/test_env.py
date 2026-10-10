from environments.car_env import CarEnv

import numpy as np


env = CarEnv()


observation, info = env.reset()

print("Initial observation:")
print(observation)


for i in range(100):

    action = np.array([
        0.5,
        0.5
    ], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    print(
        "Step:",
        i,
        "Position:",
        observation[:3],
        "Reward:",
        reward
    )

    if terminated or truncated:
        break


print("\nEnvironment test completed.")
