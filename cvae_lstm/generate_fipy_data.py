
import os
import numpy as np
import scipy.io
from fipy import Grid2D, CellVariable, DiffusionTerm, TransientTerm
from tqdm import tqdm

def generate_sequences(input_dir, output_dir, num_sequences=5, steps=50, dt=10.0):
    os.makedirs(output_dir, exist_ok=True)
    
    # List all patches
    patch_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])
    if not patch_files:
        print(f"No patch files found in {input_dir}")
        return

    # Randomly select seeds
    selected_seeds = np.random.choice(patch_files, num_sequences, replace=False)
    
    nx, ny = 80, 80
    dx = dy = 1.0
    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

    for i, seed_file in enumerate(selected_seeds):
        print(f"\nGenerating sequence {i+1}/{num_sequences} from seed: {seed_file}")
        
        # Load initial condition
        seed_path = os.path.join(input_dir, seed_file)
        mat_data = scipy.io.loadmat(seed_path)
        initial_phase = mat_data['phase']
        
        phi = CellVariable(name="phase", mesh=mesh, value=initial_phase.flatten())
        
        # Allen-Cahn style: dphi/dt = M * grad^2(phi) - L * phi*(1-phi)*(1-2*phi)
        # Simplified diffusion with strong coefficient for visible motion
        eq = TransientTerm() == DiffusionTerm(coeff=5.0)
        
        sequence = np.zeros((steps, 1, nx, ny), dtype=np.float32)
        sequence[0, 0] = initial_phase
        
        for t in tqdm(range(1, steps)):
            eq.solve(var=phi, dt=dt)
            phi.value = np.clip(phi.value, 0.0, 1.0)
            sequence[t, 0] = phi.value.reshape(nx, ny)
            
        output_path = os.path.join(output_dir, f"sequence_{i:03d}.npy")
        np.save(output_path, sequence)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_sequences(
        input_dir="zenodo_data",
        output_dir="data",
        num_sequences=5,
        steps=50,
        dt=20.0 # High dt for significant evolution
    )
