import numpy as np

from environments.car_env import CarEnv


class GhostCarEnv(CarEnv):
    """
    Ghost version of the existing CarEnv.

    Important:
    - Uses the exact same reward system as CarEnv.
    - Uses the exact same observation space.
    - Uses the exact same action space.
    - Each GhostCarEnv has its own MuJoCo simulation.
    - Therefore ghost cars cannot collide with each other.
    """

    def __init__(self, ghost_id=0):

        self.ghost_id = ghost_id

        # Use the original CarEnv
        super().__init__()

    def reset(self, seed=None, options=None):

        observation, info = super().reset(
            seed=seed,
            options=options
        )

        # Add ghost information
        info["ghost_id"] = self.ghost_id

        return observation, info

    def step(self, action):

        observation, reward, terminated, truncated, info = super().step(
            action
        )

        # Identify which ghost generated this experience
        info["ghost_id"] = self.ghost_id

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    def close(self):

        super().close()
