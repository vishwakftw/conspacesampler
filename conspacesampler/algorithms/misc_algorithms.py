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
from ..potentials import QuadraticPotential
from ..utils import (
    compute_bounds_box,
    compute_bounds_polytope,
    compute_bounds_ellipsoid,
)

__all__ = ["HitAndRunSampler", "PolytopeTruncGaussian"]


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

    def set_initial_particles(
        self, particles: torch.Tensor, check_particles: bool = True
    ):
        if particles.shape[0] != self.num_samples:
            raise ValueError(
                "Initialisation doesn't contain the same number "
                "of particles as expected."
            )
        assert torch.all(
            self.barrier.feasibility(particles)
        ), "Some initial particles are infeasible"
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


class PolytopeTruncGaussian:
    """
    Sampling from a Gaussian distribution truncated to a polytope
    using Wall Hamiltonian Monte Carlo (Pakman and Paninski, 2014).

    This implementation is based on the Matlab implementation of
    this method by Ari Pakman: https://github.com/aripakman/hmc-tmg.
    """

    WHITENED = False

    def __init__(
        self,
        barrier: PolytopeBarrier,
        potential: QuadraticPotential,
        num_samples: int,
    ):
        self.barrier = barrier
        self.potential = potential
        self.num_samples = num_samples

    def set_initial_particles(
        self, particles: torch.Tensor, check_particles: bool = True
    ):
        if particles.shape[0] != self.num_samples:
            raise ValueError(
                "Initialisation doesn't contain the same number "
                "of particles as expected."
            )
        assert torch.all(
            self.barrier.feasibility(particles)
        ), "Some initial particles are infeasible"
        self.initial_particles = particles.clone()

    def whiten(self):
        if self.WHITENED:
            return  # don't do anything
        chol_factor = torch.linalg.cholesky(self.potential.Q, upper=True)
        mean_gauss = torch.linalg.solve_triangular(
            chol_factor.T,
            torch.linalg.solve_triangular(
                chol_factor, self.potential.r.unsqueeze(-1), upper=True
            ),
            upper=False,
        ).squeeze(-1)
        # below are the polytope preprocessing steps
        b = self.barrier.polytope["b"] - self.barrier.polytope["A"] @ mean_gauss
        A = torch.linalg.solve_triangular(
            chol_factor, self.barrier.polytope["A"], left=False, upper=True
        )
        self.chol_factor = chol_factor
        self.mean_gauss = mean_gauss
        self.barrier = PolytopeBarrier({"A": A, "b": b})
        self.WHITENED = True

    def unwhiten(self):
        old_A = self.barrier.polytope["A"] @ self.chol_factor
        old_b = self.barrier.polytope["b"] + old_A @ self.mean_gauss
        self.barrier = PolytopeBarrier({"A": old_A, "b": old_b})
        self.WHITENED = False

    def _one_step(self, particles_curr: torch.Tensor, with_numeric_check: bool = True):
        velocity_curr = torch.randn_like(particles_curr)
        if with_numeric_check:
            # reflect_idxs: constraint index for each particle that
            # has been previously reflected on
            # reflect_idxs_map: mask for particles indices for which
            # reflect_idxs is defined i.e., not -1
            reflect_idxs_map = torch.zeros(self.num_samples, dtype=torch.bool)
            reflect_idxs = torch.full((self.num_samples,), -1, dtype=torch.long)

        # integration time stuff
        MAXTIME = torch.pi / 2
        total_time = torch.zeros(self.num_samples)
        actv_parts = torch.ones(self.num_samples, dtype=torch.bool)

        x, v = particles_curr.clone(), velocity_curr.clone()
        Asq = torch.sum(torch.square(self.barrier.polytope["A"]), dim=-1, keepdim=True)

        while actv_parts.any():
            # Ax, Av are both of size N x m
            # only do computation over active particles
            Ax = -self.barrier._Ax(x=x[actv_parts])
            Av = -self.barrier._Ax(x=v[actv_parts])

            # key functions
            U = torch.sqrt(torch.square(Ax) + torch.square(Av))
            phi = torch.atan2(-Av, Ax)
            b_by_U = self.barrier.polytope["b"] / U

            # check for potential boundary hitting
            # size N x m
            check = b_by_U.abs().le(1.0)

            # now we compute the integration time
            # for particles that have hit potentially
            # hit the boundary, more sophistication is required
            cand_t = -phi + torch.acos(-self.barrier.polytope["b"] / U)
            cand_t[~check] = torch.inf

            if with_numeric_check:
                # check particles for which reflect_idx is defined
                # if there are no previous reflected indices, this
                # can be safely ignored
                ref_idxs_map_actv = reflect_idxs_map[actv_parts]
                ref_idxs_actv = reflect_idxs[actv_parts]
                if ref_idxs_map_actv.any():
                    # check if the new reflection is
                    # at the same particle
                    # for each particle and the corresponding index
                    # check if this is currently hit
                    ref_again = check[ref_idxs_map_actv, ref_idxs_actv]
                    # if for no particle the previous reflected index
                    # is reflected again, then skip
                    # else do some computation
                    if ref_again.any():
                        check_cumsum = torch.cumsum(check, dim=-1) - 1  # zero-indexing
                        # for each i in reflect_idxs_actv, you get one value
                        tmp_idx = check_cumsum[ref_idxs_map_actv, ref_idxs_actv]
                        tt1 = cand_t[ref_idxs_map_actv, tmp_idx]
                        numeric = torch.isclose(tt1, torch.tensor(0.0)).bitwise_or_(
                            torch.isclose(tt1, torch.tensor(2 * torch.pi))
                        )
                        if numeric.any():
                            cand_t[numeric, tmp_idx[numeric]] = torch.inf

            delta_t, index_t = torch.min(cand_t, dim=-1)

            if with_numeric_check:
                check_any = check.any(dim=-1)
                reflect_idxs_map[actv_parts] = check_any
                reflect_idxs[reflect_idxs_map.bitwise_and(actv_parts)] = index_t[
                    check_any
                ]

            delta_t.clamp_max_(
                max=MAXTIME
            )  # when the check fails for a particle, the whole row is inf
            delta_t.add_(total_time[actv_parts]).clamp_max_(max=MAXTIME).sub_(
                total_time[actv_parts]
            )
            total_time[actv_parts] += delta_t

            # after delta_t computation, there are some particles that
            # could become inactive, and the velocity has to be updated
            # for the rest, the "rest" is captured by actv_in_group
            # actv_in_group is the subset of particles that are active
            # amongst those that were active in this iteration
            actv_in_group = total_time[actv_parts] < MAXTIME

            sin_delta_t = torch.sin(delta_t).unsqueeze_(-1)
            cos_delta_t = torch.cos(delta_t).unsqueeze_(-1)
            x_new = x[actv_parts] * cos_delta_t + v[actv_parts] * sin_delta_t
            v_new = -x[actv_parts] * sin_delta_t + v[actv_parts] * cos_delta_t
            x[actv_parts] = x_new

            # new active particles
            actv_parts = total_time < MAXTIME

            # velocity update for active particles only
            j = index_t[actv_in_group]
            Aj = self.barrier.polytope["A"][j]  # |actv_parts| x d
            qj = torch.sum(Aj * v_new[actv_in_group], dim=-1, keepdim=True) / Asq[j]
            v[actv_parts] = v_new[actv_in_group] - 2 * qj * Aj

        if with_numeric_check:
            # check feasibility just in case
            infeas = self.barrier.feasibility(x).logical_not_()
            x[infeas] = particles_curr[infeas]

        return x

    def mix(
        self,
        num_iters: int,
        no_progress: bool = True,
        return_particles: bool = True,
        with_numeric_check: bool = True,
    ):
        self.whiten()  # whiten the distribution

        particles = self.initial_particles.clone()
        particles = torch.einsum(
            "...j,ij->...i", particles - self.mean_gauss, self.chol_factor
        )
        particles_tracker = [particles]

        for _ in tqdm(range(num_iters), disable=no_progress):
            particles = self._one_step(
                particles_curr=particles, with_numeric_check=with_numeric_check
            )
            if return_particles:
                particles_tracker.append(particles)
        if return_particles:
            particles_tracker = torch.stack(particles_tracker)
            particles_tracker = (
                torch.linalg.solve_triangular(
                    self.chol_factor.T, particles_tracker.unsqueeze(-1), upper=False
                ).squeeze(-1)
                + self.mean_gauss
            )
            return particles_tracker
        else:
            return (
                torch.linalg.solve_triangular(
                    self.chol_factor.T, particles.unsqueeze(-1)
                ).squeeze(-1)
                + self.mean_gauss
            )
