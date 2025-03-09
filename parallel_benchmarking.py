import genesis as gs
import torch
import numpy as np
import cProfile
import pstats
import time
import os

########################## settings ##########################
output_file_name = "simulation_og_report.txt"
num_simulations = 10
num_environments = 50
total_steps = 1250
run_cProfile = False

########################## init ##########################
# Lists to store timing information
simulation_times = []
fps_values = []

if run_cProfile:
    cProfile.run('gs.init(backend=gs.cuda)', 'cProfile_init')
else:
    gs.init(backend=gs.cuda)

def run_hard_reset(franka, dofs_idx, scene):
    # Hard reset
    print('Running hard reset...')
    for i in range(150):
        if i < 50:
            franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
        elif i < 100:
            franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
        else:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)
        scene.step()
    print('Hard reset complete')

def run_simulation(franka, dofs_idx, scene):
    # Run a single simulation and return timing information
    simulation_start = time.time()
    
    for i in range(total_steps):
        if i == 0:
            franka.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 250:
            franka.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 500:
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        elif i == 750:
            # control first dof with velocity, and the rest with position
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
                dofs_idx[1:],
            )
            franka.control_dofs_velocity(
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
                dofs_idx[:1],
            )
        elif i == 1000:
            franka.control_dofs_force(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        
        if run_cProfile:
            cProfile.run('scene.step()', f'cProfile_step_sim{sim_num}_step{i}')
        else:
            scene.step()
    
    simulation_time = time.time() - simulation_start
    fps = total_steps / simulation_time

    return simulation_time, fps        

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer = False,
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

########################## build & run ##########################

# create B parallel environments
B = num_environments
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# Run simulations the specified number of times
for sim_num in range(num_simulations):    
    print(f"\n=== Starting Simulation Run {sim_num+1}/{num_simulations} ===")

    # Run hard reset before each simulation
    run_hard_reset(franka, dofs_idx, scene)
    
    # Run the simulation and record timing
    sim_time, fps = run_simulation(franka, dofs_idx, scene)
    
    # Store results
    simulation_times.append(sim_time)
    fps_values.append(fps)

print(f"\n=== Ended All {num_simulations} Simulation Runs ===")

########################## closing ##########################
# Calculate statistics
avg_time = np.mean(simulation_times)
std_time = np.std(simulation_times)
avg_fps = np.mean(fps_values)
std_fps = np.std(fps_values)

# Print summary
print("\n=== Simulation Summary ===")
print(f"Number of environments: {num_environments}")
print(f"Number of simulation runs: {num_simulations}")
print(f"Steps per simulation: {total_steps}")
print(f"Average simulation time: {avg_time:.6f} seconds (± {std_time:.6f})")
print(f"Average FPS: {avg_fps:.2f} (± {std_fps:.2f})")

with open(output_file_name, "w") as f:
    f.write(f"=== {output_file_name} ===\n\n")
    f.write(f"Configuration:\n")
    f.write(f"- Number of environments: {num_environments}\n")
    f.write(f"- Number of simulation runs: {num_simulations}\n")
    f.write(f"- Steps per simulation: {total_steps}\n\n")
    
    f.write(f"Summary Statistics:\n")
    f.write(f"- Average simulation time: {avg_time:.6f} seconds (± {std_time:.6f})\n")
    f.write(f"- Average FPS: {avg_fps:.2f} (± {std_fps:.2f})\n\n")
    
    f.write(f"Individual Run Results:\n")
    for i, (time_val, fps_val) in enumerate(zip(simulation_times, fps_values)):
        f.write(f"- Run {i+1}: {time_val:.6f} seconds ({fps_val:.2f} FPS)\n")

print(f'Completed write to {output_file_name}')