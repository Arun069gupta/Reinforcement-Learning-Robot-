import time
import numpy as np
import mujoco
import mujoco.viewer

from environments.car_env import CarEnv


# ==========================================================
# SETTINGS
# ==========================================================

STEPS_PER_TEST = 250
PAUSE_BETWEEN_TESTS = 1.0


# ==========================================================
# CREATE ENVIRONMENT
# ==========================================================

env = CarEnv()

print()
print("==========================================")
print("MUJOCO CAR CONTROL TEST")
print("==========================================")
print()
print("Action format:")
print("[left_wheel, right_wheel]")
print()
print("Straight : [1.0, 1.0]")
print("Left     : [0.3, 1.0]")
print("Right    : [1.0, 0.3]")
print()


# ==========================================================
# VIEWER
# ==========================================================

viewer = mujoco.viewer.launch_passive(
    env.model,
    env.data
)

viewer.cam.distance = 10
viewer.cam.azimuth = 90
viewer.cam.elevation = -30


# ==========================================================
# YAW FUNCTION
# ==========================================================

def get_yaw():

    q = env.data.qpos[3:7]

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )

    return float(yaw)


# ==========================================================
# RUN ONE TEST
# ==========================================================

def run_test(name, action):

    print()
    print("==========================================")
    print(name)
    print("Action:", action)
    print("==========================================")

    observation, info = env.reset()

    start_x = float(env.data.qpos[0])
    start_y = float(env.data.qpos[1])
    start_yaw = get_yaw()

    print(
        f"Start position: "
        f"[{start_x:.3f}, {start_y:.3f}]"
    )

    print(
        f"Start yaw: "
        f"{np.degrees(start_yaw):.2f} deg"
    )

    for step in range(STEPS_PER_TEST):

        obs, reward, terminated, truncated, info = env.step(
            np.array(action, dtype=np.float32)
        )

        viewer.sync()

        if step % 25 == 0:

            x = float(env.data.qpos[0])
            y = float(env.data.qpos[1])
            yaw = get_yaw()

            vx = float(env.data.qvel[0])
            vy = float(env.data.qvel[1])

            print(
                f"Step {step:3d} | "
                f"Pos [{x: .3f}, {y: .3f}] | "
                f"Yaw {np.degrees(yaw): .2f} deg | "
                f"Vel [{vx: .3f}, {vy: .3f}]"
            )

        time.sleep(0.005)

        if terminated or truncated:

            print("Episode ended early.")

            break

    # ------------------------------------------------------
    # FINAL STATE
    # ------------------------------------------------------

    final_x = float(env.data.qpos[0])
    final_y = float(env.data.qpos[1])
    final_yaw = get_yaw()

    yaw_change = np.degrees(
        ((final_yaw - start_yaw + np.pi) % (2.0 * np.pi))
        - np.pi
    )

    distance_moved = np.sqrt(
        (final_x - start_x) ** 2
        +
        (final_y - start_y) ** 2
    )

    print()
    print("RESULT")
    print("------------------------------------------")
    print(f"Start position : [{start_x:.3f}, {start_y:.3f}]")
    print(f"Final position : [{final_x:.3f}, {final_y:.3f}]")
    print(f"Distance moved : {distance_moved:.3f} m")
    print(f"Yaw change     : {yaw_change:.2f} deg")
    print("------------------------------------------")

    time.sleep(PAUSE_BETWEEN_TESTS)


# ==========================================================
# TEST 1 — STRAIGHT
# ==========================================================

run_test(
    "TEST 1 — STRAIGHT",
    [1.0, 1.0]
)


# ==========================================================
# TEST 2 — LEFT
# ==========================================================

run_test(
    "TEST 2 — LEFT TURN",
    [0.3, 1.0]
)


# ==========================================================
# TEST 3 — RIGHT
# ==========================================================

run_test(
    "TEST 3 — RIGHT TURN",
    [1.0, 0.3]
)


# ==========================================================
# FINISH
# ==========================================================

print()
print("==========================================")
print("ALL CONTROL TESTS COMPLETE")
print("==========================================")

viewer.close()
env.close()
