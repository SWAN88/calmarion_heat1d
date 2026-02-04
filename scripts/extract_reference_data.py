#!/usr/bin/env python3
"""
Extract reference data from Fortran output for validation testing.

Reads CSV files from src/output/ and saves as .npy files in tests/reference_data/
"""

import numpy as np
from pathlib import Path


def extract_solution(csv_path: Path) -> dict:
    """Extract solution data from Fortran CSV output."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=4)
    return {
        'x': data[:, 0],
        'u_numerical': data[:, 1],
        'u_analytical': data[:, 2],
        'error': data[:, 3]
    }


def extract_history(csv_path: Path) -> dict:
    """Extract history data from Fortran CSV output."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    return {
        'step': data[:, 0].astype(int),
        'time': data[:, 1],
        'l2_error': data[:, 2],
        'max_error': data[:, 3],
        'total_heat': data[:, 4]
    }


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'src' / 'output'
    ref_dir = base_dir / 'tests' / 'reference_data'

    ref_dir.mkdir(parents=True, exist_ok=True)

    # Extract final solution (step 250)
    solution = extract_solution(output_dir / 'solution_000250.csv')
    np.save(ref_dir / 'x.npy', solution['x'])
    np.save(ref_dir / 'u_final.npy', solution['u_numerical'])
    np.save(ref_dir / 'u_analytical_final.npy', solution['u_analytical'])

    # Extract initial solution (step 0)
    solution_init = extract_solution(output_dir / 'solution_000000.csv')
    np.save(ref_dir / 'u_initial.npy', solution_init['u_numerical'])

    # Extract intermediate solution (step 100)
    solution_100 = extract_solution(output_dir / 'solution_000100.csv')
    np.save(ref_dir / 'u_step100.npy', solution_100['u_numerical'])

    # Extract history
    history = extract_history(output_dir / 'history.csv')
    np.save(ref_dir / 'history_steps.npy', history['step'])
    np.save(ref_dir / 'history_times.npy', history['time'])
    np.save(ref_dir / 'history_l2_errors.npy', history['l2_error'])
    np.save(ref_dir / 'history_max_errors.npy', history['max_error'])
    np.save(ref_dir / 'history_total_heat.npy', history['total_heat'])

    # Save parameters as JSON for reference
    params = {
        'nx': 101,
        'L': 1.0,
        'alpha': 0.01,
        'cfl': 0.4,
        'dx': 0.01,
        'dt': 0.004,
        't_end': 1.0,
        'n_steps': 250,
        'output_freq': 100
    }

    import json
    with open(ref_dir / 'params.json', 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Reference data extracted to {ref_dir}")
    print(f"Files created:")
    for f in sorted(ref_dir.glob('*')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
