import genesis as gs
import torch
import numpy as np
import cProfile
import pstats
import time
import os

########################## settings ##########################
output_file_name = "simulation_optDRAFT_report.txt"
file_notes = ""
num_environments = 2500
num_simulations = 10
total_steps = 1250
write_report = False
run_cProfile_init = False
run_cProfile_step = True
run_CUDA_timing = False
run_visualization = False

########################## init ##########################
# Lists to store timing information
simulation_times = []
fps_values = []

if run_cProfile_init:
    profiler = cProfile.Profile()
    profiler.enable()
    import genesis as gs
    gs.init(backend=gs.cuda)
    profiler.disable()
    profiler.dump_stats('cProfile_init')
else:
    gs.init(backend=gs.cuda)

def run_hard_reset(franka, dofs_idx, scene):
    # Hard reset with batched operations
    print('Running hard reset...')
    
    # Create batched tensors for reset
    pos1 = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04], device='cuda', dtype=torch.float32)
    pos1_batch = pos1.unsqueeze(0).expand(B, -1)
    
    pos2 = torch.tensor([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04], device='cuda', dtype=torch.float32)
    pos2_batch = pos2.unsqueeze(0).expand(B, -1)
    
    pos3 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda', dtype=torch.float32)
    pos3_batch = pos3.unsqueeze(0).expand(B, -1)
    
    for i in range(150):
        if i < 50:
            franka.set_dofs_position(pos1_batch, dofs_idx)
        elif i < 100:
            franka.set_dofs_position(pos2_batch, dofs_idx)
        else:
            franka.set_dofs_position(pos3_batch, dofs_idx)
        scene.step()
    print('Hard reset complete')

def run_simulation(franka, dofs_idx, scene):
    # Run a single simulation and return timing information
    simulation_start = time.time()
    
    if run_CUDA_timing:
        # Set up CUDA timing events
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    
    # Create tensors once before the loop
    pos_tensor1 = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04], device='cuda', dtype=torch.float32)
    pos_tensor2 = torch.tensor([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04], device='cuda', dtype=torch.float32)
    pos_tensor3 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda', dtype=torch.float32)
    vel_tensor = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda', dtype=torch.float32)
    
    if run_CUDA_timing:
        # Track cumulative times
        total_control_time = 0
        total_step_time = 0
    
    for i in range(total_steps):
        if run_CUDA_timing:
            # Time control operations
            torch.cuda.synchronize()
            start.record()
        
        if i == 0:
            franka.control_dofs_position(pos_tensor1, dofs_idx)
        elif i == 250:
            franka.control_dofs_position(pos_tensor2, dofs_idx)
        elif i == 500:
            franka.control_dofs_position(pos_tensor3, dofs_idx)
        elif i == 750:
            # control first dof with velocity, and the rest with position
            franka.control_dofs_position(pos_tensor3[1:], dofs_idx[1:])
            franka.control_dofs_velocity(vel_tensor[:1], dofs_idx[:1])
        elif i == 1000:
            franka.control_dofs_force(pos_tensor3, dofs_idx)
        
        if run_CUDA_timing:
            end.record()
            torch.cuda.synchronize()
            control_time = start.elapsed_time(end)
            total_control_time += control_time
        
            # Only print detailed timing for specific steps to avoid console spam
            if i in [0, 250, 500, 750, 1000]:
                print(f"Step {i}: Control call took: {control_time:.2f} ms")
            
            # Time step operations
            torch.cuda.synchronize()
            start.record()
        
        if run_cProfile_step and i == 0: # Only cProfile the 1st step
            profiler = cProfile.Profile()
            profiler.enable()
            scene.step()
            profiler.disable()
            profiler.dump_stats(f'cProfile_step')
        else:
            scene.step()
        
        if run_CUDA_timing:
            end.record()
            torch.cuda.synchronize()
            step_time = start.elapsed_time(end)
            total_step_time += step_time
            
            # Only print detailed timing for specific steps
            if i in [0, 250, 500, 750, 1000]:
                print(f"Step {i}: Scene step took: {step_time:.2f} ms")
     
    if run_CUDA_timing:
        # Report overall timing
        print(f"\nTiming Summary:")
        print(f"Total control time: {total_control_time:.2f} ms (Average: {total_control_time/total_steps:.2f} ms per step)")
        print(f"Total step time: {total_step_time:.2f} ms (Average: {total_step_time/total_steps:.2f} ms per step)")
        print(f"Control operations: {(total_control_time/(total_control_time+total_step_time))*100:.1f}% of total time")
        print(f"Step operations: {(total_step_time/(total_control_time+total_step_time))*100:.1f}% of total time")
    
    simulation_time = time.time() - simulation_start
    fps = total_steps / simulation_time

    return simulation_time, fps  

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer = run_visualization,
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

if write_report:
    with open(output_file_name, "w") as f:
        f.write(f"=== {output_file_name} ===\n\n")
        f.write(f"Notes:\n")
        f.write(f"{file_notes}\n\n")
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