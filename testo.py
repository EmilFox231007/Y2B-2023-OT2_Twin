import numpy as np
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from simple_pid import PID  

# ✅ Plate Position (Top-Left Corner in Robot Coordinates)
plate_top_left = np.array([0.10775, 0.062, 0.057])  # (x, y, z)
plate_size_mm = 150  # Plate size in mm
plate_size_meters = plate_size_mm / 1000  # Convert mm to meters

# ✅ Plate Boundaries (Ensuring Pipette Stays on the Plate)
PLATE_MIN_X = plate_top_left[0]  # Left boundary
PLATE_MAX_X = plate_top_left[0] + plate_size_meters  # Right boundary
PLATE_MIN_Y = plate_top_left[1] - plate_size_meters  # Bottom boundary
PLATE_MAX_Y = plate_top_left[1]  # Top boundary
PLATE_Z = plate_top_left[2]  # Fixed Z height

# ✅ Define goal space limits (New Plate-Constrained Values)
GOAL_SPACE_LOW = np.array([PLATE_MIN_X, PLATE_MIN_Y, PLATE_Z])
GOAL_SPACE_HIGH = np.array([PLATE_MAX_X, PLATE_MAX_Y, PLATE_Z])

# ✅ Pixel-to-Meter Conversion
# ✅ Pixel-to-Meter Conversion
plate_size_pixels = 2604  # Plate size in pixels
plate_size_mm = 150  # Plate size in mm
conversion_factor = plate_size_mm / plate_size_pixels  # mm per pixel

# ✅ Root tip pixel coordinates (X, Y in pixels)
root_tip_pixel = np.array([1000, 1000])  # Target location on the plate

# ✅ Convert pixel coordinates to mm (No flipping Y)
root_tip_mm = np.array([
    root_tip_pixel[0] * conversion_factor,  # Convert X from pixels to mm
    root_tip_pixel[1] * conversion_factor,  # Convert Y from pixels to mm (No flip)
    0  # No Z movement
])

# ✅ Convert to meters & offset by plate_top_left
goal_position = plate_top_left + (root_tip_mm / 1000)  # Convert mm to meters

# ✅ Ensure it's within the plate boundaries
goal_position = np.clip(goal_position, GOAL_SPACE_LOW, GOAL_SPACE_HIGH)

print("\n✅ Debugging: Coordinate Transformations")
print(f"Pixel Coordinates: {root_tip_pixel}")
print(f"Converted to mm: {root_tip_mm}")
print(f"Final Goal in Robot Coordinates (meters): {goal_position}")
print(f"Plate Boundaries: X[{PLATE_MIN_X}, {PLATE_MAX_X}], Y[{PLATE_MIN_Y}, {PLATE_MAX_Y}], Z={PLATE_Z}")


# ✅ Initialize the environment
env = OT2Env(render=True)
state, _ = env.reset()
pipette_position = state[:3]  # Extract initial pipette position

# ✅ Optimized PID Controllers
pid_x = PID(100.0, 0.1, 5.0, setpoint=goal_position[0], output_limits=(-1, 1))
pid_y = PID(100.0, 0.1, 5.0, setpoint=goal_position[1], output_limits=(-1, 1))
pid_z = PID(100.0, 0.1, 5.0, setpoint=goal_position[2], output_limits=(-1, 1))

# ✅ Movement tuning
pid_scale = 0.5  # Scale the PID outputs
error_threshold = 0.0005

# ✅ Lists for visualization
time_steps = []
x_positions, y_positions, z_positions = [], [], []
setpoint_x, setpoint_y, setpoint_z = [], [], []

# ✅ Move the pipette towards the goal
for step in range(300):
    current_x, current_y, current_z = pipette_position

    # ✅ Compute PID control signals (scaled appropriately)
    control_x = pid_x(current_x) * pid_scale
    control_y = pid_y(current_y) * pid_scale
    control_z = pid_z(current_z) * pid_scale

    action = np.array([control_x, control_y, control_z])

    # ✅ Apply action and update state
    state, _, _, _, _ = env.step(action)
    pipette_position = state[:3]  # Update the pipette position from the new state

    # ✅ Compute error (distance to goal)
    distance = np.linalg.norm(goal_position - pipette_position)

    # ✅ Store data for visualization
    time_steps.append(step)
    x_positions.append(current_x)
    y_positions.append(current_y)
    z_positions.append(current_z)
    setpoint_x.append(goal_position[0])
    setpoint_y.append(goal_position[1])
    setpoint_z.append(goal_position[2])

    # ✅ Debugging output
    print(f"Step {step}: X={current_x:.5f}, Y={current_y:.5f}, Z={current_z:.5f}, "
          f"Control X={control_x:.5f}, Y={control_y:.5f}, Z={control_z:.5f}, "
          f"Goal: {goal_position}, Error={distance:.5f}")

    # ✅ Check if pipette reached the goal
    if distance < error_threshold:
        print(f"✅ Goal {goal_position} reached in {step} steps!")

        # ✅ Drop inoculum
        action = np.array([0, 0, 0, 1])  # Last value triggers inoculation
        state, _, _, _, _ = env.step(action)
        break  # ✅ Exit loop

env.close()

# ✅ Ensure Matplotlib updates properly
plt.ioff()  # Turn off interactive mode

# # ✅ Generate plots for the final movement
# def plot_position_vs_setpoint(time_steps, positions, setpoints, axis_name):
#     """ Plots position vs setpoint for a single axis. """
#     plt.figure(figsize=(8, 4))
#     plt.xlabel("Time Steps")
#     plt.ylabel(f"{axis_name} Position")
#     plt.plot(time_steps, positions, label=f"{axis_name} Position", color="red")
#     plt.plot(time_steps, setpoints, label=f"{axis_name} Setpoint", linestyle="dashed", color="blue")
#     plt.title(f"{axis_name}-Axis Position Tracking")
#     plt.legend()
#     plt.grid(True)
#     plt.show(block=True)  # ✅ Ensure plots stay visible

# # ✅ Plot final trajectory
# plot_position_vs_setpoint(time_steps, x_positions, setpoint_x, "X")
# plot_position_vs_setpoint(time_steps, y_positions, setpoint_y, "Y")
# plot_position_vs_setpoint(time_steps, z_positions, setpoint_z, "Z")

# print("✅ Evaluation complete. Plots generated.")
