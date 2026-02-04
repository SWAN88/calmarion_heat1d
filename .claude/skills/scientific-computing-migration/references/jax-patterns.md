# JAX Optimization Patterns

## Basic JIT Usage

```python
from jax import jit
import jax.numpy as jnp

# Basic JIT
@jit
def compute(x, y):
    return jnp.dot(x, y) + jnp.sin(x)

# JIT with static arguments (for values that change shape/behavior)
from functools import partial

@partial(jit, static_argnums=(2,))
def compute_with_mode(x, y, mode):
    if mode == 'add':
        return x + y
    else:
        return x * y
```

## Vectorization with vmap

### Basic vmap

```python
from jax import vmap

def single_item_func(x):
    """Operates on a single item."""
    return jnp.sum(x ** 2)

# Vectorize over first axis
batched_func = vmap(single_item_func)

# Vectorize over specific axis
batched_func = vmap(single_item_func, in_axes=0)  # Batch over axis 0
batched_func = vmap(single_item_func, in_axes=1)  # Batch over axis 1
```

### Nested vmap for Pairwise Operations

```python
def pairwise_distance(x_i, x_j):
    """Distance between two points."""
    return jnp.linalg.norm(x_i - x_j)

# All pairwise distances
# Result shape: (N, N) for N points
all_distances = vmap(vmap(pairwise_distance, in_axes=(None, 0)), in_axes=(0, None))

positions = jnp.array([[0, 0], [1, 0], [0, 1]])  # 3 points
distances = all_distances(positions, positions)  # (3, 3) matrix
```

### vmap with Multiple Inputs

```python
def weighted_sum(x, w):
    return jnp.sum(x * w)

# Batch over x (axis 0), broadcast w
batched = vmap(weighted_sum, in_axes=(0, None))

# Batch over both
batched = vmap(weighted_sum, in_axes=(0, 0))
```

### vmap Output Axes

```python
def func(x):
    return x, x ** 2  # Returns tuple

# Control output axes
batched = vmap(func, in_axes=0, out_axes=(0, 1))
```

## Loop Optimization with scan

### Basic scan

```python
from jax.lax import scan

def step_fn(carry, x):
    """
    Args:
        carry: State passed between iterations
        x: Current input from xs sequence
    Returns:
        new_carry: Updated state
        y: Output for this step (stacked into ys)
    """
    state = carry
    new_state = state + x
    output = new_state ** 2
    return new_state, output

init_state = 0.0
xs = jnp.array([1.0, 2.0, 3.0, 4.0])
final_state, outputs = scan(step_fn, init_state, xs)
# final_state = 10.0
# outputs = [1, 9, 36, 100]
```

### scan Without Inputs (Fixed Iterations)

```python
def simulation_step(state, _):
    positions, velocities = state
    # Update physics
    accelerations = compute_accelerations(positions)
    new_velocities = velocities + accelerations * dt
    new_positions = positions + new_velocities * dt
    return (new_positions, new_velocities), None

initial_state = (init_positions, init_velocities)
final_state, _ = scan(simulation_step, initial_state, None, length=1000)
```

### scan with Output Collection

```python
def step_with_output(state, _):
    positions, velocities = state
    energy = compute_energy(positions, velocities)
    
    accelerations = compute_accelerations(positions)
    new_velocities = velocities + accelerations * dt
    new_positions = positions + new_velocities * dt
    
    return (new_positions, new_velocities), energy

final_state, energy_history = scan(step_with_output, initial_state, None, length=1000)
# energy_history has shape (1000,)
```

### scan with Multiple Outputs

```python
def step_multi_output(state, _):
    positions, velocities = state
    
    energy = compute_energy(positions, velocities)
    momentum = compute_momentum(velocities)
    
    new_state = update_state(positions, velocities)
    
    return new_state, (energy, momentum, positions)

final_state, (energies, momenta, trajectories) = scan(
    step_multi_output, initial_state, None, length=1000
)
```

## Control Flow

### Conditional Execution with cond

```python
from jax.lax import cond

def true_fn(x):
    return x ** 2

def false_fn(x):
    return x ** 3

# cond evaluates only one branch
result = cond(condition, true_fn, false_fn, operand)

# With multiple operands
def true_fn(args):
    x, y = args
    return x + y

def false_fn(args):
    x, y = args
    return x * y

result = cond(condition, true_fn, false_fn, (x, y))
```

### Simple Selection with select

```python
from jax.lax import select

# select: both branches evaluated (use for simple values)
result = select(condition, true_value, false_value)

# For arrays, jnp.where is often clearer
result = jnp.where(condition_array, true_values, false_values)
```

### Switch Statement

```python
from jax.lax import switch

def case_0(x):
    return x

def case_1(x):
    return x ** 2

def case_2(x):
    return x ** 3

# index selects which function to call
result = switch(index, [case_0, case_1, case_2], operand)
```

### While Loops

```python
from jax.lax import while_loop

def cond_fn(state):
    """Return True to continue looping."""
    x, i = state
    return i < 100

def body_fn(state):
    """Loop body."""
    x, i = state
    return (x + 0.1, i + 1)

initial_state = (0.0, 0)
final_state = while_loop(cond_fn, body_fn, initial_state)
```

### For Loops with fori_loop

```python
from jax.lax import fori_loop

def body_fn(i, state):
    """Body receives loop index and state."""
    return state + i

initial_state = 0
final_state = fori_loop(0, 10, body_fn, initial_state)
# Equivalent to: for i in range(0, 10): state = body_fn(i, state)
```

## Precision Control

```python
# Ensure 64-bit precision globally (must be set before importing jax.numpy)
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Explicit dtype specification
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

# Check precision
print(x.dtype)  # Should be float64

# Convert existing arrays
x_f64 = x.astype(jnp.float64)
```

## Debugging

### Debug Printing Inside JIT

```python
from jax import debug

@jit
def buggy_function(x):
    intermediate = x ** 2
    debug.print("intermediate: {}", intermediate)  # Works inside JIT
    debug.print("shape: {s}, dtype: {d}", s=intermediate.shape, d=intermediate.dtype)
    return intermediate + 1
```

### Inspect Traced Computation

```python
from jax import make_jaxpr

def my_func(x, y):
    return jnp.dot(x, y) + jnp.sin(x)

# See the JAX intermediate representation
print(make_jaxpr(my_func)(jnp.ones(3), jnp.ones((3, 3))))
```

### Disable JIT for Debugging

```python
import jax

# Context manager
with jax.disable_jit():
    result = my_jitted_function(x)
    # Now you can use print(), pdb, etc.

# Or globally (for debugging session)
jax.config.update("jax_disable_jit", True)
```

### Check for NaN/Inf

```python
from jax import config
config.update("jax_debug_nans", True)

# Now JAX will raise an error when NaN is produced
```

## Memory Management

### Gradient Checkpointing

```python
from jax import checkpoint

# Checkpoint to reduce memory (recomputes during backward pass)
@checkpoint
def memory_intensive_layer(x):
    # Many operations that use memory
    y = jnp.dot(x, weights1)
    y = jnp.tanh(y)
    y = jnp.dot(y, weights2)
    return y
```

### Explicit Device Placement

```python
from jax import device_put, device_get, devices

# List available devices
print(devices())  # [GpuDevice(id=0), ...]

# Put data on specific device
x_gpu = device_put(x, devices('gpu')[0])

# Transfer back to CPU (as NumPy array)
x_cpu = device_get(x_gpu)
```

### Managing Device Memory

```python
# Clear cached computations
from jax import clear_caches
clear_caches()

# For large arrays, process in batches
def process_in_batches(data, batch_size):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        result = jitted_process(batch)
        results.append(device_get(result))  # Move to CPU
    return jnp.concatenate(results)
```

## Random Numbers

```python
from jax import random

# JAX requires explicit key management
key = random.PRNGKey(42)

# Split key for multiple uses
key, subkey1, subkey2 = random.split(key, 3)

# Generate random numbers
uniform = random.uniform(subkey1, shape=(100,))
normal = random.normal(subkey2, shape=(100,))
integers = random.randint(key, shape=(10,), minval=0, maxval=100)

# In loops, split key each iteration
def stochastic_step(carry, _):
    state, key = carry
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=state.shape)
    new_state = state + noise * 0.1
    return (new_state, key), None

# Random permutation
shuffled_indices = random.permutation(key, jnp.arange(100))
```

## Advanced Patterns

### Automatic Differentiation

```python
from jax import grad, value_and_grad

# Gradient of scalar function
def loss(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

grad_fn = grad(loss)  # Returns gradient w.r.t. first argument
gradients = grad_fn(params, x, y)

# Get both value and gradient
loss_and_grad = value_and_grad(loss)
loss_value, gradients = loss_and_grad(params, x, y)
```

### Jacobian and Hessian

```python
from jax import jacobian, hessian

def f(x):
    return jnp.array([x[0]**2 + x[1], x[0] * x[1]])

J = jacobian(f)(jnp.array([1.0, 2.0]))  # 2x2 Jacobian matrix

def g(x):
    return jnp.sum(x ** 3)

H = hessian(g)(jnp.array([1.0, 2.0, 3.0]))  # 3x3 Hessian matrix
```

### Custom Derivatives

```python
from jax import custom_vjp

@custom_vjp
def safe_sqrt(x):
    return jnp.sqrt(x)

def safe_sqrt_fwd(x):
    y = jnp.sqrt(x)
    return y, (x, y)  # Save for backward

def safe_sqrt_bwd(res, g):
    x, y = res
    # Custom gradient that handles x=0
    grad = jnp.where(x > 0, g / (2 * y), 0.0)
    return (grad,)

safe_sqrt.defvjp(safe_sqrt_fwd, safe_sqrt_bwd)
```

### Parallelization with pmap

```python
from jax import pmap

# Parallelize across multiple devices (GPUs/TPUs)
@pmap
def parallel_compute(x):
    return jnp.dot(x, x.T)

# Input must have leading axis matching number of devices
n_devices = jax.device_count()
x = jnp.ones((n_devices, 100, 100))
result = parallel_compute(x)  # Runs on all devices
```

## Common Pitfalls

### 1. Avoid Python Control Flow in JIT

```python
# BAD: Python if inside JIT
@jit
def bad_func(x):
    if x > 0:  # This traces based on concrete value at first call
        return x
    return -x

# GOOD: Use JAX control flow
@jit
def good_func(x):
    return jnp.where(x > 0, x, -x)

# Or use cond for more complex branches
@jit
def good_func_cond(x):
    return cond(x > 0, lambda x: x, lambda x: -x, x)
```

### 2. Avoid Dynamic Shapes

```python
# BAD: Shape depends on values
@jit
def bad_func(x):
    mask = x > 0
    return x[mask]  # Dynamic shape!

# GOOD: Keep shape static
@jit
def good_func(x):
    return jnp.where(x > 0, x, 0.0)  # Same shape as input
```

### 3. Avoid Side Effects

```python
# BAD: Side effects
results = []
@jit
def bad_func(x):
    results.append(x)  # Side effect - only happens during tracing!
    return x ** 2

# GOOD: Return all needed values
@jit
def good_func(x):
    return x ** 2, x  # Return intermediate if needed
```

### 4. Avoid In-Place Updates

```python
# BAD: In-place modification
@jit
def bad_func(x):
    x[0] = 1.0  # JAX arrays are immutable!
    return x

# GOOD: Use functional updates
@jit
def good_func(x):
    return x.at[0].set(1.0)

# Multiple updates
@jit
def good_func_multi(x):
    return x.at[0].set(1.0).at[1].add(2.0)
```

### 5. JIT Recompilation

```python
# BAD: Causes recompilation on every call with different shapes
@jit
def bad_func(x):
    return x ** 2

bad_func(jnp.ones(10))   # Compiles for shape (10,)
bad_func(jnp.ones(20))   # Recompiles for shape (20,)!

# GOOD: Pad to fixed shape or use static_argnums
@partial(jit, static_argnums=(1,))
def good_func(x, shape):
    return x[:shape] ** 2
```

### 6. Random Number Handling

```python
# BAD: Reusing same key
key = random.PRNGKey(0)
a = random.normal(key, (10,))
b = random.normal(key, (10,))  # Same as a!

# GOOD: Split keys
key = random.PRNGKey(0)
key, key_a, key_b = random.split(key, 3)
a = random.normal(key_a, (10,))
b = random.normal(key_b, (10,))  # Different from a
```

## Performance Tips

### 1. Fuse Operations

```python
# Multiple JIT calls have overhead
result = jit(f1)(x)
result = jit(f2)(result)
result = jit(f3)(result)

# Better: Fuse into single JIT
@jit
def fused(x):
    return f3(f2(f1(x)))
```

### 2. Use vmap Instead of Loops

```python
# Slow: Python loop
def slow(xs):
    return jnp.array([f(x) for x in xs])

# Fast: vmap
fast = vmap(f)
```

### 3. Prefer Static Shapes

JAX optimizes better when shapes are known at compile time.

### 4. Warm-up JIT

```python
# First call compiles
_ = my_jitted_func(dummy_input)

# Subsequent calls are fast
result = my_jitted_func(real_input)
```

### 5. Profile with JAX Profiler

```python
import jax.profiler

with jax.profiler.trace("/tmp/jax-trace"):
    result = my_func(x)

# View with: tensorboard --logdir=/tmp/jax-trace
```
