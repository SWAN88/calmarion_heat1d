# Fortran to Python Conversion Patterns

## Data Types

| Fortran | Python/NumPy |
|---------|--------------|
| INTEGER | int / np.int32 |
| INTEGER*8 | np.int64 |
| REAL | np.float32 |
| REAL*8, DOUBLE PRECISION | np.float64 |
| COMPLEX | np.complex64 |
| COMPLEX*16 | np.complex128 |
| LOGICAL | bool / np.bool_ |
| CHARACTER(LEN=N) | str |

## Array Declaration

```fortran
REAL*8 :: A(100, 200)        ! Static allocation
REAL*8, ALLOCATABLE :: B(:,:) ! Dynamic allocation
ALLOCATE(B(N, M))
```

```python
A = np.zeros((100, 200), dtype=np.float64)  # Static
B = np.zeros((N, M), dtype=np.float64)      # Dynamic
```

## Array Indexing

Fortran uses 1-based indexing and column-major order.
Python uses 0-based indexing and row-major order.

```fortran
! Fortran: A(row, col) with column-major storage
DO j = 1, NCOLS
    DO i = 1, NROWS
        A(i, j) = i + j
    END DO
END DO
```

```python
# Option 1: Keep Fortran ordering, adjust indices
for j in range(NCOLS):
    for i in range(NROWS):
        A[i, j] = (i + 1) + (j + 1)

# Option 2: Transpose to Python conventions
# A has shape (NCOLS, NROWS) in Python
for i in range(NROWS):
    for j in range(NCOLS):
        A[j, i] = (i + 1) + (j + 1)

# Option 3: Vectorized (preferred)
i_idx, j_idx = np.meshgrid(np.arange(1, NROWS+1), 
                            np.arange(1, NCOLS+1), 
                            indexing='ij')
A = i_idx + j_idx
```

## COMMON Blocks

```fortran
COMMON /PARAMS/ dt, n_particles, box_size
```

```python
# Option 1: Module-level variables (not recommended)
dt = None
n_particles = None
box_size = None

# Option 2: Configuration class (recommended)
from dataclasses import dataclass

@dataclass
class SimulationParams:
    dt: float
    n_particles: int
    box_size: float

params = SimulationParams(dt=0.001, n_particles=1000, box_size=10.0)

# Option 3: Pass as arguments (most explicit)
def simulate(positions, velocities, dt, n_particles, box_size):
    ...
```

## Subroutine vs Function

```fortran
SUBROUTINE update_positions(x, v, dt, n)
    INTEGER, INTENT(IN) :: n
    REAL*8, INTENT(INOUT) :: x(n), v(n)
    REAL*8, INTENT(IN) :: dt
    x = x + v * dt
END SUBROUTINE

FUNCTION compute_energy(v, m, n) RESULT(energy)
    INTEGER, INTENT(IN) :: n
    REAL*8, INTENT(IN) :: v(n), m(n)
    REAL*8 :: energy
    energy = 0.5 * SUM(m * v**2)
END FUNCTION
```

```python
def update_positions(x, v, dt):
    """Modifies x in place."""
    x += v * dt
    # or return x + v * dt for immutable version

def compute_energy(v, m):
    """Returns scalar energy."""
    return 0.5 * np.sum(m * v**2)
```

## Control Flow

### IF-THEN-ELSE

```fortran
IF (x > 0) THEN
    y = SQRT(x)
ELSE IF (x == 0) THEN
    y = 0
ELSE
    y = -SQRT(-x)
END IF
```

```python
if x > 0:
    y = np.sqrt(x)
elif x == 0:
    y = 0
else:
    y = -np.sqrt(-x)

# Vectorized version
y = np.where(x > 0, np.sqrt(np.abs(x)), 
             np.where(x == 0, 0, -np.sqrt(np.abs(x))))
```

### DO Loops

```fortran
DO i = 1, N, 2  ! Start, end, step
    A(i) = i**2
END DO
```

```python
for i in range(0, N, 2):  # 0-indexed
    A[i] = (i + 1)**2

# Vectorized
indices = np.arange(0, N, 2)
A[indices] = (indices + 1)**2
```

### GOTO and Labels

```fortran
      DO 100 i = 1, N
          IF (A(i) < 0) GOTO 200
          B(i) = SQRT(A(i))
100   CONTINUE
      GOTO 300
200   PRINT *, 'Negative value found'
300   CONTINUE
```

```python
found_negative = False
for i in range(N):
    if A[i] < 0:
        found_negative = True
        break
    B[i] = np.sqrt(A[i])

if found_negative:
    print('Negative value found')
```

## Intrinsic Functions

| Fortran | NumPy |
|---------|-------|
| ABS(x) | np.abs(x) |
| SQRT(x) | np.sqrt(x) |
| EXP(x) | np.exp(x) |
| LOG(x) | np.log(x) |
| LOG10(x) | np.log10(x) |
| SIN(x), COS(x), TAN(x) | np.sin(x), np.cos(x), np.tan(x) |
| ASIN(x), ACOS(x), ATAN(x) | np.arcsin(x), np.arccos(x), np.arctan(x) |
| ATAN2(y, x) | np.arctan2(y, x) |
| SINH(x), COSH(x), TANH(x) | np.sinh(x), np.cosh(x), np.tanh(x) |
| MOD(a, b) | np.mod(a, b) or a % b |
| MAX(a, b) | np.maximum(a, b) |
| MIN(a, b) | np.minimum(a, b) |
| MAXVAL(A) | np.max(A) |
| MINVAL(A) | np.min(A) |
| SUM(A) | np.sum(A) |
| PRODUCT(A) | np.prod(A) |
| DOT_PRODUCT(A, B) | np.dot(A, B) |
| MATMUL(A, B) | np.matmul(A, B) or A @ B |
| TRANSPOSE(A) | A.T or np.transpose(A) |
| SIZE(A, DIM) | A.shape[DIM-1] |
| SHAPE(A) | A.shape |
| FLOOR(x) | np.floor(x) |
| CEILING(x) | np.ceil(x) |
| NINT(x) | np.rint(x).astype(int) |
| SIGN(a, b) | np.copysign(a, b) |

## Array Operations

### Array Slicing

```fortran
A(1:10)           ! Elements 1 through 10
A(1:10:2)         ! Elements 1, 3, 5, 7, 9
A(:, 1)           ! First column
A(1, :)           ! First row
```

```python
A[0:10]           # Elements 0 through 9
A[0:10:2]         # Elements 0, 2, 4, 6, 8
A[:, 0]           # First column
A[0, :]           # First row
```

### Array Reshaping

```fortran
B = RESHAPE(A, [M, N])
```

```python
B = A.reshape(M, N)
# Note: Fortran reshapes column-major, NumPy row-major
# For exact equivalence:
B = A.reshape(N, M, order='F').T
```

### WHERE Statement

```fortran
WHERE (A > 0)
    B = SQRT(A)
ELSEWHERE
    B = 0
END WHERE
```

```python
B = np.where(A > 0, np.sqrt(A), 0)
```

## I/O Operations

### Formatted Write

```fortran
WRITE(*, '(A, I5, F10.3)') 'Step:', n, energy
WRITE(10, '(3E20.12)') x, y, z
```

```python
print(f'Step: {n:5d} {energy:10.3f}')
with open('output.dat', 'w') as f:
    f.write(f'{x:20.12e} {y:20.12e} {z:20.12e}\n')
```

### Formatted Read

```fortran
READ(10, '(3E20.12)') x, y, z
```

```python
with open('input.dat', 'r') as f:
    line = f.readline()
    x, y, z = map(float, line.split())
```

### Binary I/O

```fortran
OPEN(10, FILE='data.bin', FORM='UNFORMATTED', ACCESS='STREAM')
WRITE(10) array
CLOSE(10)
```

```python
# NumPy binary (recommended for Python interop)
np.save('data.npy', array)

# Raw binary (for Fortran compatibility)
array.astype(np.float64).tofile('data.bin')

# Reading Fortran unformatted (with record markers)
# Fortran adds 4-byte record length markers
import struct
with open('data.bin', 'rb') as f:
    rec_len = struct.unpack('i', f.read(4))[0]
    data = np.frombuffer(f.read(rec_len), dtype=np.float64)
    rec_len_end = struct.unpack('i', f.read(4))[0]
```

### Fortran Unformatted Sequential

```fortran
OPEN(10, FILE='data.bin', FORM='UNFORMATTED')
WRITE(10) n_particles
WRITE(10) positions
WRITE(10) velocities
CLOSE(10)
```

```python
import scipy.io

# Using scipy's FortranFile for proper handling
from scipy.io import FortranFile

# Reading
f = FortranFile('data.bin', 'r')
n_particles = f.read_ints(dtype=np.int32)[0]
positions = f.read_reals(dtype=np.float64).reshape(-1, 3)
velocities = f.read_reals(dtype=np.float64).reshape(-1, 3)
f.close()

# Writing
f = FortranFile('output.bin', 'w')
f.write_record(np.array([n_particles], dtype=np.int32))
f.write_record(positions.flatten().astype(np.float64))
f.write_record(velocities.flatten().astype(np.float64))
f.close()
```

## Module Structure

```fortran
MODULE physics_module
    IMPLICIT NONE
    
    REAL*8, PARAMETER :: G = 6.67430e-11
    REAL*8, PARAMETER :: PI = 3.14159265358979323846
    
    REAL*8, ALLOCATABLE :: positions(:,:)
    REAL*8, ALLOCATABLE :: velocities(:,:)
    
CONTAINS
    
    SUBROUTINE initialize(n)
        INTEGER, INTENT(IN) :: n
        ALLOCATE(positions(3, n))
        ALLOCATE(velocities(3, n))
    END SUBROUTINE
    
    FUNCTION compute_energy() RESULT(energy)
        REAL*8 :: energy
        ! Implementation
    END FUNCTION
    
END MODULE
```

```python
# physics_module.py
import numpy as np

# Constants
G = 6.67430e-11
PI = 3.14159265358979323846

# Module-level state (use with caution)
positions = None
velocities = None

def initialize(n):
    global positions, velocities
    positions = np.zeros((n, 3), dtype=np.float64)
    velocities = np.zeros((n, 3), dtype=np.float64)

def compute_energy():
    # Implementation
    pass

# Alternative: Object-oriented approach (recommended)
class PhysicsSimulation:
    G = 6.67430e-11
    PI = 3.14159265358979323846
    
    def __init__(self, n):
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.velocities = np.zeros((n, 3), dtype=np.float64)
    
    def compute_energy(self):
        # Implementation
        pass
```

## OpenMP Parallelization

```fortran
!$OMP PARALLEL DO PRIVATE(i, j, r_vec, r_mag, f_mag)
DO i = 1, N
    DO j = i+1, N
        r_vec = positions(:, j) - positions(:, i)
        r_mag = SQRT(SUM(r_vec**2))
        f_mag = G * masses(i) * masses(j) / r_mag**2
        ! Update forces
    END DO
END DO
!$OMP END PARALLEL DO
```

```python
# Option 1: NumPy vectorization (often faster than explicit parallelism)
# See JAX patterns for GPU acceleration

# Option 2: Numba with parallel=True
from numba import jit, prange

@jit(nopython=True, parallel=True)
def compute_forces_parallel(positions, masses, G):
    N = len(masses)
    forces = np.zeros_like(positions)
    for i in prange(N):
        for j in range(i + 1, N):
            r_vec = positions[j] - positions[i]
            r_mag = np.sqrt(np.sum(r_vec**2))
            f_mag = G * masses[i] * masses[j] / r_mag**2
            f_vec = f_mag * r_vec / r_mag
            forces[i] += f_vec
            forces[j] -= f_vec
    return forces

# Option 3: multiprocessing
from multiprocessing import Pool

def process_particle(args):
    i, positions, masses, G = args
    # Compute forces for particle i
    return forces_i

with Pool() as pool:
    results = pool.map(process_particle, 
                       [(i, positions, masses, G) for i in range(N)])
```

## Common Pitfalls

### 1. Index Off-by-One Errors

Always remember Fortran is 1-indexed, Python is 0-indexed.

### 2. Memory Layout

Fortran: Column-major (first index varies fastest)
NumPy: Row-major by default (last index varies fastest)

```python
# Force Fortran memory order if needed
A = np.zeros((M, N), dtype=np.float64, order='F')
```

### 3. Integer Division

```fortran
I = 5 / 2  ! Result: 2 (integer division)
```

```python
i = 5 // 2  # Result: 2 (explicit integer division)
i = 5 / 2   # Result: 2.5 (float division in Python 3)
```

### 4. Array Assignment

```fortran
A = B  ! Creates a copy
```

```python
A = B        # Creates a reference (same memory!)
A = B.copy() # Creates a copy
A = np.array(B)  # Also creates a copy
```

### 5. Implicit Type Conversion

Fortran may implicitly convert types. Python/NumPy can too, but be explicit:

```python
# Ensure float64 precision
result = np.float64(a) * np.float64(b)
```
