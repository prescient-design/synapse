#!/usr/bin/env python3
"""Launch wandb sweeps."""

import re
import subprocess
import sys

SWEEPS = {"model": "wandb_sweeps/model_comparison_sweep.yaml",
          "transfer": "wandb_sweeps/transfer_learning_sweep.yaml"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SWEEPS:
        print(f"Usage: python run_sweep.py [{'/'.join(SWEEPS.keys())}]")
        sys.exit(1)

    config = SWEEPS[sys.argv[1]]
    print(f"Initializing: {config}")
    
    result = subprocess.run(["wandb", "sweep", config], capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    match = re.search(r'wandb agent ([^\s]+)', output) or re.search(r'sweep/([a-zA-Z0-9]+)', output)
    if not match:
        print(f"Failed. Run manually: wandb sweep {config}")
        sys.exit(1)

    sweep_id = match.group(1)
    print(f"Starting agent: {sweep_id}")
    subprocess.run(["wandb", "agent", sweep_id])


if __name__ == "__main__":
    main()
