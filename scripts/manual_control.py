import mujoco
import mujoco.viewer
import pygame


# ==========================
# LOAD MUJOCO MODEL
# ==========================

model = mujoco.MjModel.from_xml_path("models/car.xml")
data = mujoco.MjData(model)


# ==========================
# INITIALIZE PYGAME
# ==========================

pygame.init()

pygame.display.set_mode((400, 200))
pygame.display.set_caption("RL Car Controller")

clock = pygame.time.Clock()


# ==========================
# MOTOR SETTINGS
# ==========================

MOTOR_FORCE = 1.5


# ==========================
# START MUJOCO VIEWER
# ==========================

with mujoco.viewer.launch_passive(model, data) as viewer:

    running = True

    while running:

        # --------------------------
        # Keyboard events
        # --------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False


        # --------------------------
        # Read keyboard
        # --------------------------

        keys = pygame.key.get_pressed()

        left_motor = 0.0
        right_motor = 0.0


        # Forward
        if keys[pygame.K_w]:
            left_motor = MOTOR_FORCE
            right_motor = MOTOR_FORCE


        # Reverse
        elif keys[pygame.K_s]:
            left_motor = -MOTOR_FORCE
            right_motor = -MOTOR_FORCE


        # Turn left
        elif keys[pygame.K_a]:
            left_motor = -MOTOR_FORCE
            right_motor = MOTOR_FORCE


        # Turn right
        elif keys[pygame.K_d]:
            left_motor = MOTOR_FORCE
            right_motor = -MOTOR_FORCE


        # --------------------------
        # Apply motor commands
        # --------------------------

        data.ctrl[0] = left_motor
        data.ctrl[1] = right_motor


        # --------------------------
        # Physics
        # --------------------------

        mujoco.mj_step(model, data)

        viewer.sync()

        clock.tick(100)


pygame.quit()
