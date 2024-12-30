# conspacesampler

A (work-in-progress) repository for constrained sampling methods i.e., algorithms for sampling from continuous distributions whose support is a proper convex subset of a Euclidean space.
This is a spin-off from the [`metromirrorlangevin`](https://github.com/vishwakftw/metropolis-adjusted-MLA) package in an attempt to collate implementations of practical constrained sampling methods.

The algorithms implemented here are all based on `PyTorch`.

### How to use `conspacesampler`?

With `setup.py`, it is as easy as running `pip install .` (or `pip install -e .` if you would like to work on it by yourself).
Below is an example:

```python
import torch

from conspacesampler import algorithms, barriers

# Define a log-barrier for a 2D box [-0.01, 0.01] x [1, 1]
barrier = barriers.BoxBarrier(bounds=torch.tensor([0.01, 1]))

# Define the sampler instance, with number of samples = 500
sampler = algorithms.misc_algorithms.HitAndRunSampler(
    barrier=barrier,
    num_samples=500
)

# Initialise the particles
# in the smaller box [-0.001, 0.001] x [-0.001, 0.001]
sampler.set_initial_particles(torch.rand(500, 2) * 0.002 - 0.001)

# Perform the mixing for 1000 iterations, with step size 0.05
# particles is of shape (num_iters, num_samples, dimension)
# rejects is of shape (num_iters, num_samples)
particles = sampler.mix(
    num_iters=1000,
    stepsize=0.05,
    return_particles=True,
    no_progress=False
)
```

Documentation will be updated continuously, and will be made available soon!