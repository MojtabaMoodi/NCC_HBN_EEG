#!/usr/bin/env python3
"""
Generate SLURM job scripts for each training script in the runs directory.
For Compute Canada/Niagara servers.
"""

import os
from pathlib import Path

# SLURM configuration for Compute Canada/Niagara
SLURM_CONFIG = {
    'account': 'aip-aghodsib',  # Your Compute Canada account
    'time': '24:00:00',  # 24 hours (adjust as needed)
    'nodes': 1,
    'ntasks_per_node': 1,
    'cpus_per_task': 10,  # Match num_workers in scripts
    'mem': '64G',  # Adjust based on your needs
    'gres': 'gpu:h100:1',  # 1 H100 GPU per job
    'job_name_template': 'labram_{script_name}',
    'output_template': './logs/slurm/%x-%j.out',
    'error_template': './logs/slurm/%x-%j.err',
}

# Module setup for Compute Canada/Niagara
MODULE_SETUP = """# Load required modules
module load python/3.10
module load cuda/12.6
module load cudnn/8.9.7

# Initialize conda
source ~/.bashrc || true
eval "$(conda shell.bash hook)" || true

# Activate labram conda environment
conda activate labram
"""

def get_job_name(script_name):
    """Extract job name from script filename."""
    # Remove .sh extension and 'run_' prefix
    name = script_name.replace('.sh', '').replace('run_', '')
    return f"labram_{name}"

def estimate_time(script_name):
    """Estimate job time based on script type."""
    if 'cv' in script_name:
        # CV experiments take much longer (5 folds) - use 3-7 days
        if 'cross_task_cv' in script_name:
            return '7-00:00:00'  # Cross-task CV is most intensive (7 days)
        else:
            return '3-00:00:00'  # Regular CV (3 days)
    elif 'baseline' in script_name:
        return '1-00:00:00'  # Baseline experiments (1 day)
    elif 'cross_task' in script_name:
        return '1-00:00:00'  # Cross-task experiments (1 day)
    else:
        return '1-00:00:00'  # Default (1 day)

def estimate_memory(script_name):
    """Estimate memory requirements based on script type."""
    if '4s' in script_name:
        return '64G'  # 4s segments need more memory
    elif 'cross_task_cv' in script_name and '4s' in script_name:
        return '128G'  # Most memory intensive
    else:
        return '32G'

def create_slurm_job(script_path, output_dir):
    """Create a SLURM job script for a given training script."""
    script_name = script_path.name
    job_name = get_job_name(script_name)
    time_limit = estimate_time(script_name)
    memory = estimate_memory(script_name)
    
    # Get relative path from runs directory to script
    script_rel_path = script_path.name
    
    # Create SLURM job script content
    slurm_content = f"""#!/bin/bash
#SBATCH --account={SLURM_CONFIG['account']}
#SBATCH --time={time_limit}
#SBATCH --nodes={SLURM_CONFIG['nodes']}
#SBATCH --ntasks-per-node={SLURM_CONFIG['ntasks_per_node']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --mem={memory}
#SBATCH --gres={SLURM_CONFIG['gres']}
#SBATCH --job-name={job_name}
#SBATCH --output={SLURM_CONFIG['output_template']}
#SBATCH --error={SLURM_CONFIG['error_template']}

# Job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Print system information
echo "System Information:"
echo "Hostname: $(hostname)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep Mem | awk '{{print $2}}')"
echo ""

{MODULE_SETUP}
# Change to LaBraM directory (runs directory is in LaBraM/runs)
cd "$(dirname "$0")/.."

# Run the training script
echo "=========================================="
echo "Starting training: {script_rel_path}"
echo "=========================================="
echo ""

bash runs/{script_rel_path}

echo ""
echo "=========================================="
echo "Job completed: {script_rel_path}"
echo "End Time: $(date)"
echo "=========================================="
"""
    
    # Write SLURM job script
    output_path = output_dir / f"{script_name.replace('.sh', '.slurm')}"
    with open(output_path, 'w') as f:
        f.write(slurm_content)
    
    # Make it executable
    os.chmod(output_path, 0o755)
    
    return output_path

def main():
    """Generate SLURM job scripts for all training scripts."""
    runs_dir = Path(__file__).parent
    output_dir = runs_dir / 'slurm_jobs'
    output_dir.mkdir(exist_ok=True)
    
    # Create logs directory for SLURM output
    logs_dir = runs_dir.parent / 'logs' / 'slurm'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .sh scripts
    scripts = sorted(runs_dir.glob('run_*.sh'))
    
    print(f"Found {len(scripts)} training scripts")
    print(f"Generating SLURM job scripts in: {output_dir}")
    print("")
    
    generated = []
    for script in scripts:
        slurm_script = create_slurm_job(script, output_dir)
        generated.append(slurm_script)
        print(f"Generated: {slurm_script.name}")
    
    print("")
    print(f"Generated {len(generated)} SLURM job scripts")
    print("")
    print("To submit all jobs, run:")
    print(f"  cd {output_dir}")
    print("  for script in *.slurm; do sbatch $script; done")
    print("")
    print("Or submit individually:")
    print(f"  cd {output_dir}")
    print("  sbatch <script_name>.slurm")

if __name__ == '__main__':
    main()

