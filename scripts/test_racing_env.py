from environments.car_env import CarEnv

import numpy as np


env = CarEnv()


observation, info = env.reset()


print("\n==============================")
print("RACING ENVIRONMENT TEST")
print("==============================")

print("\nInitial observation:")
print(observation)

print("\nInitial info:")
print(info)


for step in range(100):

    # Move both wheels forward

    action = np.array(
        [0.5, 0.5],
        dtype=np.float32
    )


    observation, reward, terminated, truncated, info = env.step(
        action
    )


    if step % 10 == 0:

        print(
            f"\nStep: {step}"
        )

        print(
            f"Progress: "
            f"{info['progress']:.4f}"
        )

        print(
            f"Distance: "
            f"{info['distance_from_centerline']:.4f}"
        )

        print(
            f"Velocity: "
            f"{info['forward_velocity']:.4f}"
        )

        print(
            f"Reward: "
            f"{reward:.4f}"
        )


    if terminated or truncated:

        print("\nEpisode ended.")

        break


print("\n==============================")
print("TEST COMPLETED")
print("==============================")
