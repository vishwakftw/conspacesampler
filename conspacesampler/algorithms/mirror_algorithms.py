import math
import torch

torch.set_default_dtype(torch.float64)

from tqdm.auto import tqdm

from ..barriers import Barrier
from ..potentials import Potential
from ..utils import get_chol

__all__ = [
    "UniformMMRWSampler",
    "GeneralMAMLASampler",
    "MirrorLangevinSampler",
]


class UniformMMRWSampler:
    """
    Metropolis Mirror Random Walk
    """

    def __init__(
        self,
        barrier: Barrier,
        num_samples: int,
    ):
        self.barrier = barrier
        self.num_samples = num_samples
        self.CHOL = get_chol(is_diagonal=barrier.diag_hess)

    def set_initial_particles(self, particles: torch.Tensor):
        # NOTE: these are particles in the primal space, not the dual space.
        if particles.shape[0] != self.num_samples:
            raise ValueError(
                "Initialisation doesn't contain the same number "
                "of particles as expected."
            )
        self.initial_particles = particles.clone()

    def _compute_log_proposal_prob_ratio(
        self,
        noise: torch.Tensor,
        scaled_noise: torch.Tensor,
        chol_cov_curr: torch.Tensor,
        chol_cov_prop: torch.Tensor,
    ):
        # scaled_noise is Sqrt[cov] @ noise
        if self.barrier.diag_hess:
            # when hessian is diagonal
            # computation can be simplified
            # logdet is just sum of logs
            logdet_term = 3 * (
                torch.log(chol_cov_curr).sum(dim=-1)
                - torch.log(chol_cov_prop).sum(dim=-1)
            )
            # tmp_quantity is inv(Lz) @ Lx @ \xi = inv(Lz) @ s\xi
            # Lz is n x d
            tmp_quantity = (
                scaled_noise / chol_cov_prop
            )  # this is because chol_cov_prop is a diagonal
        else:
            # when hessian is not diagonal
            # some computation is necessary
            # logdet is just sum of logs of the diagonals
            logdet_term = 3 * (
                torch.diagonal(chol_cov_curr, dim1=-2, dim2=-1).log().sum(dim=-1)
                - torch.diagonal(chol_cov_prop, dim1=-2, dim2=-1).log().sum(dim=-1)
            )
            # tmp_quantity is inv(Lz) @ Lx @ \xi = inv(Lz) @ s\xi
            # Lz is n x d x d, and triangular
            tmp_quantity = torch.linalg.solve_triangular(
                chol_cov_prop, scaled_noise.unsqueeze(-1), upper=False
            ).squeeze(-1)

        diff_norms_term = 0.5 * (
            torch.sum(torch.square(noise), dim=-1)
            - torch.sum(torch.square(tmp_quantity), dim=-1)
        )
        return logdet_term + diff_norms_term

    def _update_particles(
        self,
        particles: torch.Tensor,
        particles_dual: torch.Tensor,
        chol_cov: torch.Tensor,
        full_cov: torch.Tensor,
        stepsize: float,
    ):
        # chol_cov: n x d if diagonal, n x d x d if not diagonal
        # full_cov: n x d if diagonal, n x d x d if not diagonal
        # proposal step
        noise = torch.randn_like(particles_dual)
        if self.barrier.diag_hess:
            # scaled_noise s\xi : Lx @ \xi
            # simpler computation
            scaled_noise = chol_cov * noise
        else:
            # scaled_noise s\xi : Lx @ \xi
            # mat-vec
            scaled_noise = torch.einsum("bij,bj->bi", chol_cov, noise)
        prop_particles_dual = particles_dual + math.sqrt(2 * stepsize) * scaled_noise
        prop_particles = self.barrier.inverse_gradient(prop_particles_dual)
        full_cov_prop = self.barrier.hessian(prop_particles)
        chol_cov_prop = self.CHOL(full_cov_prop)
        # chol_cov_prop: n x d if diagonal, n x d x d if not diagonal
        # full_cov_prop: n x d if diagonal, n x d x d if not diagonal

        # compute acceptance prob
        # log_alpha: n
        log_alpha = self._compute_log_proposal_prob_ratio(
            noise=noise,
            scaled_noise=scaled_noise,
            chol_cov_curr=chol_cov,
            chol_cov_prop=chol_cov_prop,
        )
        uniform_vals = torch.empty_like(log_alpha).uniform_().log_()
        reject = uniform_vals > log_alpha
        # reject: n
        # restore old particles
        prop_particles[reject] = particles[reject]
        prop_particles_dual[reject] = particles_dual[reject]
        chol_cov_prop[reject] = chol_cov[reject]
        full_cov_prop[reject] = full_cov[reject]
        return (
            prop_particles,
            prop_particles_dual,
            chol_cov_prop,
            full_cov_prop,
            reject,
        )

    def mix(
        self,
        num_iters: int,
        stepsize: float,
        no_progress: bool = True,
        return_particles: bool = True,
    ):
        particles = self.initial_particles.clone()
        particles_dual = self.barrier.gradient(particles)
        if return_particles:
            particles_tracker = [particles]
        reject_tracker = []

        full_covariance = self.barrier.hessian(particles)
        # if diag_hess, n x d, else n x d x d
        chol_covariance = self.CHOL(full_covariance)
        # if diag_hess, n x d, else n x d x d
        for _ in tqdm(range(num_iters), disable=no_progress):
            (
                particles,
                particles_dual,
                chol_covariance,
                full_covariance,
                reject,
            ) = self._update_particles(
                particles,
                particles_dual,
                chol_covariance,
                full_covariance,
                stepsize,
            )
            if return_particles:
                particles_tracker.append(particles)
            reject_tracker.append(reject)

        reject_tracker = torch.stack(reject_tracker)
        if return_particles:
            return torch.stack(particles_tracker), reject_tracker
        else:  # return final sample
            return particles, reject_tracker


class GeneralMAMLASampler:
    def __init__(
        self,
        barrier: Barrier,
        potential: Potential,
        num_samples: int,
    ):
        self.barrier = barrier
        self.potential = potential
        self.num_samples = num_samples
        self.CHOL = get_chol(is_diagonal=barrier.diag_hess)

    def set_initial_particles(self, particles: torch.Tensor):
        # NOTE: these are particles in the primal space, not the dual space.
        if particles.shape[0] != self.num_samples:
            raise ValueError(
                "Initialisation doesn't contain the same number "
                "of particles as expected."
            )
        self.initial_particles = particles.clone()

    def _compute_log_proposal_prob_ratio(
        self,
        noise: torch.Tensor,
        scaled_noise: torch.Tensor,
        particles: torch.Tensor,
        prop_particles: torch.Tensor,
        grad_potential_curr: torch.Tensor,
        chol_cov_curr: torch.Tensor,
        chol_cov_prop: torch.Tensor,
        stepsize: float,
    ):
        potential_diff = self.potential.value(particles) - self.potential.value(
            prop_particles
        )

        grad_potential_prop = self.potential.gradient(prop_particles)

        # scaled_noise is Sqrt[cov] @ noise
        if self.barrier.diag_hess:
            # when hessian is diagonal
            # computation can be simplified
            # logdet is just sum of logs
            logdet_term = 3 * (
                torch.log(chol_cov_curr).sum(dim=-1)
                - torch.log(chol_cov_prop).sum(dim=-1)
            )
            # tmp_quantity_1 is inv(Lz) @ Lx @ \xi = inv(Lz) @ s\xi
            # Lz is n x d
            tmp_quantity_1 = (
                scaled_noise / chol_cov_prop
            )  # this is because chol_cov_prop is a diagonal
            # tmp_quantity_2 is inv(Lz) @ (gradf(z) + gradf(x))
            tmp_quantity_2 = (grad_potential_prop + grad_potential_curr) / chol_cov_prop
        else:
            # when hessian is not diagonal
            # some computation is necessary
            # logdet is just sum of logs of the diagonals
            logdet_term = 3 * (
                torch.diagonal(chol_cov_curr, dim1=-2, dim2=-1).log().sum(dim=-1)
                - torch.diagonal(chol_cov_prop, dim1=-2, dim2=-1).log().sum(dim=-1)
            )
            # tmp_quantity_1 is inv(Lz) @ Lx @ \xi = inv(Lz) @ s\xi
            # Lz is n x d x d, and triangular
            tmp_quantity_1 = torch.linalg.solve_triangular(
                chol_cov_prop, scaled_noise.unsqueeze(-1), upper=False
            ).squeeze(-1)
            # tmp_quantity_2 is inv(Lz) @ (gradf(z) + gradf(x))
            tmp_quantity_2 = torch.linalg.solve_triangular(
                chol_cov_prop,
                (grad_potential_prop + grad_potential_curr).unsqueeze(-1),
                upper=False,
            ).squeeze(-1)

        q1 = -stepsize * torch.sum(torch.square(tmp_quantity_2), dim=-1) / 4
        q2 = 0.5 * (
            torch.sum(torch.square(noise), dim=-1)
            - torch.sum(torch.square(tmp_quantity_1), dim=-1)
        )
        q3 = math.sqrt(stepsize / 2) * torch.sum(
            tmp_quantity_1 * tmp_quantity_2, dim=-1
        )
        return q1 + q2 + q3 + potential_diff + logdet_term, grad_potential_prop

    def _update_particles(
        self,
        particles: torch.Tensor,
        particles_dual: torch.Tensor,
        grad_potential: torch.Tensor,
        chol_cov: torch.Tensor,
        full_cov: torch.Tensor,
        stepsize: float,
    ):
        # proposal step
        noise = torch.randn_like(particles_dual)
        if self.barrier.diag_hess:
            # scaled_noise s\xi : Lx @ \xi
            # simpler computation
            scaled_noise = chol_cov * noise
        else:
            # scaled_noise s\xi : Lx @ \xi
            # mat-vec
            scaled_noise = torch.einsum("bij,bj->bi", chol_cov, noise)
        prop_particles_dual = (
            particles_dual
            - stepsize * grad_potential
            + math.sqrt(2 * stepsize) * scaled_noise
        )
        prop_particles = self.barrier.inverse_gradient(prop_particles_dual)
        full_cov_prop = self.barrier.hessian(prop_particles)
        chol_cov_prop = self.CHOL(full_cov_prop)
        # chol_cov_prop: n x d if diagonal, n x d x d if not diagonal
        # full_cov_prop: n x d if diagonal, n x d x d if not diagonal

        # compute acceptance prob
        log_alpha, grad_potential_prop = self._compute_log_proposal_prob_ratio(
            noise=noise,
            scaled_noise=scaled_noise,
            particles=particles,
            prop_particles=prop_particles,
            grad_potential_curr=grad_potential,
            chol_cov_curr=chol_cov,
            chol_cov_prop=chol_cov_prop,
            stepsize=stepsize,
        )
        uniform_vals = torch.empty_like(log_alpha).uniform_().log_()
        reject = uniform_vals > log_alpha
        # reject: n
        # restore old particles
        prop_particles[reject] = particles[reject]
        prop_particles_dual[reject] = particles_dual[reject]
        grad_potential_prop[reject] = grad_potential[reject]
        chol_cov_prop[reject] = chol_cov[reject]
        full_cov_prop[reject] = full_cov[reject]
        return (
            prop_particles,
            prop_particles_dual,
            grad_potential_prop,
            chol_cov_prop,
            full_cov_prop,
            reject,
        )

    def mix(
        self,
        num_iters: int,
        stepsize: float,
        no_progress: bool = True,
        return_particles: bool = True,
    ):
        particles = self.initial_particles.clone()
        particles_dual = self.barrier.gradient(particles)
        particles_tracker = [particles]
        reject_tracker = []

        full_covariance = self.barrier.hessian(particles)
        # if diag_hess, n x d, else n x d x d
        chol_covariance = self.CHOL(full_covariance)
        # if diag_hess, n x d, else n x d x d
        grad_potential = self.potential.gradient(particles)
        for _ in tqdm(range(num_iters), disable=no_progress):
            (
                particles,
                particles_dual,
                grad_potential,
                chol_covariance,
                full_covariance,
                reject,
            ) = self._update_particles(
                particles=particles,
                particles_dual=particles_dual,
                grad_potential=grad_potential,
                chol_cov=chol_covariance,
                full_cov=full_covariance,
                stepsize=stepsize,
            )
            particles_tracker.append(particles)
            reject_tracker.append(reject)
        reject_tracker = torch.stack(reject_tracker)
        if return_particles:
            return torch.stack(particles_tracker), reject_tracker
        else:  # return final sample
            return particles, reject_tracker


class MirrorLangevinSampler:
    def __init__(
        self,
        barrier: Barrier,
        potential: Potential,
        num_samples: int,
    ):
        self.barrier = barrier
        self.potential = potential
        self.num_samples = num_samples
        self.CHOL = get_chol(is_diagonal=barrier.diag_hess)

    def set_initial_particles(self, particles: torch.Tensor):
        # NOTE: these are particles in the primal space, not the dual space.
        if particles.shape[0] != self.num_samples:
            raise ValueError(
                "Initialisation doesn't contain the same number "
                "of particles as expected."
            )
        self.initial_particles = particles.clone()

    def mix(
        self,
        num_iters: int,
        stepsize: float,
        return_particles: bool = False,
        no_progress: bool = True,
    ):
        particles = self.initial_particles.clone()
        particles_dual = self.barrier.gradient(particles)
        particles_tracker = [particles]

        full_covariance = self.barrier.hessian(particles)
        # if diag_hess, n x d, else n x d x d
        chol_covariance = self.CHOL(full_covariance)
        # if diag_hess, n x d, else n x d x d
        grad_potential = self.potential.gradient(particles)
        for _ in tqdm(range(num_iters), disable=no_progress):
            noise = torch.randn_like(particles_dual)
            if self.barrier.diag_hess:
                # scaled_noise = Lx @ xi = Lx * xi
                scaled_noise = chol_covariance * noise
            else:
                # scaled_noise = Lx @ xi
                scaled_noise = torch.einsum("bij,bj->bi", chol_covariance, noise)

            particles_dual = (
                particles_dual
                - stepsize * grad_potential
                + math.sqrt(2 * stepsize) * scaled_noise
            )
            particles = self.barrier.inverse_gradient(particles_dual)
            full_covariance = self.barrier.hessian(particles)
            chol_covariance = self.CHOL(full_covariance)
            particles_tracker.append(particles)
        if return_particles:
            return torch.stack(particles_tracker)
        else:
            return particles
