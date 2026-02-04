# CalculiX Integration Patterns

Reading, writing, and validating against CalculiX finite element results.

## Reading CalculiX Output Files

### Parse .frd Results File

```python
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class FrdResult:
    """Parsed CalculiX .frd results."""
    nodes: np.ndarray              # (n_nodes, 3) coordinates
    node_ids: np.ndarray           # (n_nodes,) node IDs
    elements: Dict[str, np.ndarray]  # connectivity by element type
    temperature: np.ndarray        # (n_nodes,) or (n_nodes, n_steps)
    time_steps: np.ndarray         # (n_steps,) time values


def parse_frd(filepath: Path) -> FrdResult:
    """
    Parse CalculiX .frd results file.
    
    Supports:
    - Nodal coordinates (block 2C)
    - Element connectivity (block 3C)
    - Temperature results (NT)
    - Displacement results (DISP)
    """
    nodes = []
    node_ids = []
    elements = {}
    temperatures = []
    time_steps = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Node coordinates block (2C)
        if line.startswith('2C'):
            n_nodes = int(lines[i][6:18])
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].split()
                node_ids.append(int(parts[0]))
                nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
                i += 1
            continue
        
        # Element connectivity block (3C)
        if line.startswith('3C'):
            n_elements = int(lines[i][6:18])
            elem_type = lines[i][18:24].strip()
            i += 1
            connectivity = []
            for _ in range(n_elements):
                parts = lines[i].split()
                connectivity.append([int(p) for p in parts[1:]])
                i += 1
            elements[elem_type] = np.array(connectivity)
            continue
        
        # Temperature results block (NT)
        if 'NT' in line and line.startswith(' -4'):
            time_val = float(lines[i-1].split()[1]) if i > 0 else 0.0
            time_steps.append(time_val)
            i += 1
            temp_vals = []
            while i < len(lines) and not lines[i].strip().startswith('-3'):
                parts = lines[i].split()
                if len(parts) >= 2:
                    temp_vals.append(float(parts[1]))
                i += 1
            temperatures.append(temp_vals)
            continue
        
        i += 1
    
    return FrdResult(
        nodes=np.array(nodes),
        node_ids=np.array(node_ids),
        elements=elements,
        temperature=np.array(temperatures).T if temperatures else np.array([]),
        time_steps=np.array(time_steps)
    )
```

### Parse .inp Mesh File

```python
def parse_inp_mesh(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse mesh from CalculiX .inp file.
    
    Returns:
        nodes: (n_nodes, 3) coordinates
        elements: (n_elements, nodes_per_element) connectivity
    """
    nodes = {}
    elements = []
    mode = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('*NODE'):
                mode = 'nodes'
                continue
            elif line.startswith('*ELEMENT'):
                mode = 'elements'
                continue
            elif line.startswith('*'):
                mode = None
                continue
            
            if mode == 'nodes' and line and not line.startswith('**'):
                parts = line.split(',')
                node_id = int(parts[0])
                coords = [float(p) for p in parts[1:4]]
                nodes[node_id] = coords
            
            elif mode == 'elements' and line and not line.startswith('**'):
                parts = line.split(',')
                elem_nodes = [int(p) for p in parts[1:]]
                elements.append(elem_nodes)
    
    sorted_ids = sorted(nodes.keys())
    node_array = np.array([nodes[i] for i in sorted_ids])
    elem_array = np.array(elements)
    
    return node_array, elem_array
```

---

## Writing CalculiX Input Files

### Generate Thermal Analysis Input

```python
def write_thermal_inp(
    filepath: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    conductivity: float,
    specific_heat: float,
    density: float,
    bc_nodes: Dict[int, float],
    initial_temp: float = 293.0,
    time_end: float = 1.0,
    dt: float = 0.01,
    steady_state: bool = True
):
    """
    Generate CalculiX input for thermal analysis.
    
    Args:
        nodes: (n_nodes, 3) coordinates
        elements: (n_elements, 8) connectivity for C3D8 elements
        bc_nodes: {node_id: temperature} for Dirichlet BC
    """
    with open(filepath, 'w') as f:
        # Header
        f.write("** CalculiX thermal analysis input\n")
        f.write("** Generated by Python\n\n")
        
        # Nodes
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(nodes, 1):
            f.write(f"{i}, {x:.10e}, {y:.10e}, {z:.10e}\n")
        
        # Elements
        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        for i, conn in enumerate(elements, 1):
            conn_str = ", ".join(str(n) for n in conn)
            f.write(f"{i}, {conn_str}\n")
        
        # Node set for all nodes
        f.write("*NSET, NSET=NALL, GENERATE\n")
        f.write(f"1, {len(nodes)}, 1\n")
        
        # Material
        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*CONDUCTIVITY\n")
        f.write(f"{conductivity}\n")
        f.write("*SPECIFIC HEAT\n")
        f.write(f"{specific_heat}\n")
        f.write("*DENSITY\n")
        f.write(f"{density}\n")
        
        # Section
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n\n")
        
        # Initial conditions
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"NALL, {initial_temp}\n\n")
        
        # Boundary conditions
        f.write("*BOUNDARY\n")
        for node_id, temp in bc_nodes.items():
            f.write(f"{node_id}, 11, 11, {temp}\n")
        
        # Step
        f.write("\n*STEP\n")
        if steady_state:
            f.write("*HEAT TRANSFER, STEADY STATE\n")
        else:
            f.write(f"*HEAT TRANSFER\n{dt}, {time_end}\n")
        
        # Output requests
        f.write("*NODE FILE\n")
        f.write("NT\n")
        f.write("*EL FILE\n")
        f.write("HFL\n")
        f.write("*END STEP\n")
```

---

## Running CalculiX from Python

```python
import subprocess

def run_calculix(inp_file: Path, ccx_path: str = "ccx") -> Path:
    """
    Run CalculiX solver.
    
    Args:
        inp_file: Path to .inp file
        ccx_path: Path to ccx executable
        
    Returns:
        Path to .frd results file
    """
    job_name = inp_file.stem
    work_dir = inp_file.parent
    
    result = subprocess.run(
        [ccx_path, job_name],
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"CalculiX failed:\n{result.stderr}")
    
    frd_file = work_dir / f"{job_name}.frd"
    if not frd_file.exists():
        raise FileNotFoundError(f"Output not found: {frd_file}")
    
    return frd_file
```

---

## Validation Against CalculiX

```python
import jax.numpy as jnp

def validate_against_calculix(
    jax_result: jnp.ndarray,
    frd_filepath: Path,
    rtol: float = 1e-10,
    atol: float = 1e-12
) -> dict:
    """
    Validate JAX solution against CalculiX results.
    
    Returns:
        Dictionary with validation metrics
    """
    ccx_result = parse_frd(frd_filepath)
    ccx_temp = ccx_result.temperature[:, -1]  # Final time step
    
    assert jax_result.shape == ccx_temp.shape, \
        f"Shape mismatch: JAX {jax_result.shape} vs CCX {ccx_temp.shape}"
    
    abs_error = jnp.abs(jax_result - ccx_temp)
    rel_error = abs_error / (jnp.abs(ccx_temp) + 1e-15)
    
    passed = jnp.allclose(jax_result, ccx_temp, rtol=rtol, atol=atol)
    
    return {
        'passed': bool(passed),
        'max_absolute_error': float(jnp.max(abs_error)),
        'max_relative_error': float(jnp.max(rel_error)),
        'mean_absolute_error': float(jnp.mean(abs_error)),
        'rtol': rtol,
        'atol': atol,
        'n_nodes': len(jax_result)
    }
```

---

## Mesh Interpolation

```python
from scipy.interpolate import griddata

def interpolate_to_regular_grid(
    nodes: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, int, int],
    bounds: tuple
) -> jnp.ndarray:
    """
    Interpolate CalculiX nodal results to regular grid.
    
    Args:
        nodes: (n_nodes, 3) coordinates
        values: (n_nodes,) nodal values
        grid_shape: (nx, ny, nz)
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    nx, ny, nz = grid_shape
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    values_interp = griddata(nodes, values, grid_points, method='linear')
    
    return jnp.array(values_interp.reshape(grid_shape))
```
