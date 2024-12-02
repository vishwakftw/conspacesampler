import torch

torch.set_default_dtype(torch.float64)

from functools import partial
from tqdm.auto import tqdm

from ..barriers import (
    Barrier,
    PolytopeBarrier,
    BoxBarrier,
    SimplexBarrier,
    EllipsoidBarrier,
)
from ..utils import (
    compute_bounds_box,
    compute_bounds_polytope,
    compute_bounds_ellipsoid,
)

__all__ = ["HitAndRunSampler"]


class HitAndRunSampler:
    def __init__(
        self,
        barrier: Barrier,
        num_samples: int,
    ):
        if isinstance(barrier, PolytopeBarrier):
            self.bounds_compute_fn = partial(compute_bounds_polytope, barrier=barrier)
        elif isinstance(barrier, SimplexBarrier):
            A = torch.cat(
                [-torch.eye(barrier.dimension), torch.ones(barrier.dimension, 1)], dim=0
            )
            b = torch.cat([torch.zeros(barrier.dimension), torch.ones(1)], dim=0)
            self.bounds_compute_fn = partial(
                compute_bounds_polytope, barrier=PolytopeBarrier({"A": A, "b": b})
            )
        elif isinstance(barrier, BoxBarrier):
            self.bounds_compute_fn = partial(compute_bounds_box, barrier=barrier)
        elif isinstance(barrier, EllipsoidBarrier):
            self.bounds_compute_fn = partial(compute_bounds_ellipsoid, barrier=barrier)
        else:
            raise NotImplementedError(
                "The current implementation does not extend beyond polytopes"
            )

        self.barrier = barrier
        self.dimension = barrier.dimension
        self.num_samples = num_samples

    def set_initial_particles(self, particles: torch.Tensor):
        if particles.shape[0] != self.num_samples:
            raise ValueError(
                "Initialisation doesn't contain the same number "
                "of particles as expected."
            )
        self.initial_particles = particles.clone()

    def _generate_directions(self, num_directions: int):
        vecs = torch.randn(num_directions, self.dimension)
        return vecs / torch.linalg.norm(vecs, dim=-1, keepdim=True)

    def mix(
        self, num_iters: int, no_progress: bool = True, return_particles: bool = True
    ):
        particles = self.initial_particles.clone()
        particles_tracker = [particles]

        for _ in tqdm(range(num_iters), disable=no_progress):
            directions = self._generate_directions(num_directions=self.num_samples)
            rlow, rhigh = self.bounds_compute_fn(
                particles=particles, directions=directions
            )
            magnitude = torch.rand_like(rlow) * (rhigh - rlow) + rlow
            update = directions * magnitude
            particles = particles + update
            if return_particles:
                particles_tracker.append(particles)
        if return_particles:
            return torch.stack(particles_tracker)
        else:
            return particles
