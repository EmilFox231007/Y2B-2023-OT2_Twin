import numpy as np
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from simple_pid import PID  

# ✅ Define goal space limits
GOAL_SPACE_LOW = np.array([-0.1904, -0.1712, -0.1205])
GOAL_SPACE_HIGH = np.array([0.255, 0.2203, 0.2906])

# ✅ Function to move the pipette to a given goal
def move_to_goal(env, goal_position, run_index, pipette_position):
    """ Moves the pipette to a given goal using optimized PID control. """

    print(f"\n🔹 Run {run_index + 1}: Moving from {pipette_position} to goal {goal_position}")

    # ✅ Optimize PID controllers for fast convergence
    pid_x = PID(200.0, 0.1, 15.0, setpoint=goal_position[0], output_limits=(-1, 1))
    pid_y = PID(200.0, 0.1, 15.0, setpoint=goal_position[1], output_limits=(-1, 1))
    pid_z = PID(200.0, 0.1, 15.0, setpoint=goal_position[2], output_limits=(-1, 1))


    # ✅ Threshold to stop when close enough
    error_threshold = 0.0005

    # ✅ Lists for storing data (only last run is kept for plotting)
    time_steps = []
    x_positions, y_positions, z_positions = [], [], []
    setpoint_x, setpoint_y, setpoint_z = [], [], []

    # ✅ Move the pipette towards the goal
    for step in range(300):  # Lower step limit for efficiency
        current_x, current_y, current_z = pipette_position

        # ✅ Compute PID control signals (scaled to appropriate range)
        control_x = pid_x(current_x) * 0.5
        control_y = pid_y(current_y) * 0.5
        control_z = pid_z(current_z) * 0.3
        drop_command = 0  # ✅ FIX: Add fourth element for the action

        # ✅ Modified action to match the expected shape (4,)
        action = np.array([control_x, control_y, control_z, drop_command])

        # ✅ Apply action and update state
        state, _, _, _, _ = env.step(action)
        pipette_position = state[:3]  # Update the pipette position from the new state

        # ✅ Compute error (distance to goal)
        distance = np.linalg.norm(goal_position - pipette_position)

        # ✅ Store data only for the last run
        if run_index == 4:  # Store only the last (5th) run for plotting
            time_steps.append(step)
            x_positions.append(current_x)
            y_positions.append(current_y)
            z_positions.append(current_z)
            setpoint_x.append(goal_position[0])
            setpoint_y.append(goal_position[1])
            setpoint_z.append(goal_position[2])

        # ✅ Debugging output
        print(f"Step {step}: X={current_x:.4f}, Y={current_y:.4f}, Z={current_z:.4f}, "
              f"Control X={control_x:.4f}, Y={control_y:.4f}, Z={control_z:.4f}, "
              f"Goal: {goal_position}, Error={distance:.6f}")

        # ✅ Check if pipette reached the goal
        if distance < error_threshold:
            print(f"✅ Goal {goal_position} reached in {step} steps!")
            return pipette_position, time_steps, x_positions, y_positions, z_positions, setpoint_x, setpoint_y, setpoint_z

    print(f"✅ Run {run_index + 1} complete.")

    # ✅ If no early exit, return the last known position
    return pipette_position, time_steps, x_positions, y_positions, z_positions, setpoint_x, setpoint_y, setpoint_z

# ✅ Initialize the environment
env = OT2Env(render=True)

# ✅ Reset environment to get the initial pipette position
state, _ = env.reset()
pipette_position = state[:3]  # Extract the initial pipette position

# ✅ Move to 5 random goal positions sequentially
for i in range(5):
    random_goal = np.random.uniform(GOAL_SPACE_LOW, GOAL_SPACE_HIGH)  # Generate a random goal within limits
    pipette_position, time_steps, x_positions, y_positions, z_positions, setpoint_x, setpoint_y, setpoint_z = move_to_goal(env, random_goal, i, pipette_position)

# ✅ Close the environment
env.close()

# ✅ Ensure Matplotlib updates properly
plt.ioff()  # Turn off interactive mode

# ✅ Generate plots for the last run
def plot_position_vs_setpoint(time_steps, positions, setpoints, axis_name):
    """ Plots position vs setpoint for a single axis. """
    plt.figure(figsize=(8, 4))
    plt.xlabel("Time Steps")
    plt.ylabel(f"{axis_name} Position")
    plt.plot(time_steps, positions, label=f"{axis_name} Position", color="red")
    plt.plot(time_steps, setpoints, label=f"{axis_name} Setpoint", linestyle="dashed", color="blue")
    plt.title(f"{axis_name}-Axis Position Tracking (Final Run)")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)  # ✅ Ensure plots stay visible

# ✅ Plot only the final run's data
plot_position_vs_setpoint(time_steps, x_positions, setpoint_x, "X")
plot_position_vs_setpoint(time_steps, y_positions, setpoint_y, "Y")
plot_position_vs_setpoint(time_steps, z_positions, setpoint_z, "Z")

print("✅ All 5 runs completed successfully. Plots generated for the final run.")
