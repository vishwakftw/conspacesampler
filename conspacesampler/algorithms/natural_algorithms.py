import math
import torch

torch.set_default_dtype(torch.float64)

from tqdm.auto import tqdm
from typing import Optional

from ..barriers import Barrier
from ..potentials import Potential
from ..utils import get_chol

__all__ = ["GeneralMAPLASampler", "GeneralDikinSampler"]


class GeneralMAPLASampler:
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
        potential_curr: torch.Tensor,
        potential_prop: torch.Tensor,
        precon_grad_potential_curr: torch.Tensor,
        precon_grad_potential_prop: torch.Tensor,
        chol_cov_curr: torch.Tensor,
        chol_cov_prop: torch.Tensor,
        stepsize: float,
    ):
        # all this computation assumes feasibility of iterates
        potential_diff = potential_curr - potential_prop

        # scaled_noise is Sqrt[inv(cov)] @ noise
        tmp_quantity_1 = (
            math.sqrt(stepsize)
            * (precon_grad_potential_curr + precon_grad_potential_prop)
            - math.sqrt(2) * scaled_noise
        )
        if self.barrier.diag_hess:
            # when hessian is diagonal
            # computation can be simplified
            # logdet is just sum of logs
            logdet_term = torch.log(chol_cov_prop).sum(dim=-1) - torch.log(
                chol_cov_curr
            ).sum(dim=-1)
            q1 = torch.sum(torch.square(tmp_quantity_1 * chol_cov_prop), dim=-1) * 0.25

        else:
            # when hessian is not diagonal
            # some computation is necessary
            # logdet is just sum of logs of the diagonals
            logdet_term = torch.diagonal(chol_cov_prop, dim1=-2, dim2=-1).log().sum(
                dim=-1
            ) - torch.diagonal(chol_cov_curr, dim1=-2, dim2=-1).log().sum(dim=-1)
            q1 = torch.einsum("bji,bj->bi", chol_cov_prop, tmp_quantity_1)
            q1 = torch.sum(torch.square(q1), dim=-1) * 0.25

        q2 = torch.sum(torch.square(noise), dim=-1) * 0.5

        return potential_diff + logdet_term - q1 + q2

    def _update_particles(
        self,
        particles: torch.Tensor,
        val_potential: torch.Tensor,
        precon_grad_potential: torch.Tensor,
        chol_cov: torch.Tensor,
        stepsize: float,
    ):
        # proposal step
        noise = torch.randn_like(particles)
        if self.barrier.diag_hess:
            # scaled_noise s\xi : Lx^{-1} * \xi
            # simpler computation
            scaled_noise = noise / chol_cov
        else:
            # scaled_noise s\xi : Lx^{-T} @ \xi
            # mat-vec
            scaled_noise = torch.linalg.solve_triangular(
                chol_cov.transpose(1, 2), noise.unsqueeze(-1), upper=True
            ).squeeze(-1)

        prop_particles = (
            particles
            - stepsize * precon_grad_potential
            + math.sqrt(2 * stepsize) * scaled_noise
        )
        # check feasibility of particles
        feasibility = self.barrier.feasibility(prop_particles)

        # particles that are infeasible will automatically
        # get rejected, so the acceptance ratio only needs to be
        # calculated for the feasible ones.
        chol_cov_prop = torch.empty_like(chol_cov)
        chol_cov_prop[feasibility] = self.CHOL(
            self.barrier.hessian(prop_particles[feasibility])
        )
        val_potential_prop_feas, grad_potential_prop_feas = (
            self.potential.value_and_gradient(prop_particles[feasibility])
        )
        val_potential_prop = torch.empty_like(val_potential)
        val_potential_prop[feasibility] = val_potential_prop_feas
        precon_grad_potential_prop = torch.empty_like(precon_grad_potential)
        if self.barrier.diag_hess:
            precon_grad_potential_prop[feasibility] = (
                grad_potential_prop_feas / chol_cov_prop[feasibility] ** 2
            )
        else:
            precon_grad_potential_prop[feasibility] = torch.cholesky_solve(
                grad_potential_prop_feas.unsqueeze(-1),
                chol_cov_prop[feasibility],
            ).squeeze(-1)

        # memory opt
        del val_potential_prop_feas, grad_potential_prop_feas

        # chol_cov_prop_feas: n' x d if hessian is diagonal, else n' x d x d
        # grad_potential_prop_feas: n' x d
        # precon_grad_potential_prop_feas: n' x d
        # where n' is the number of feasible proposals
        log_alpha_feas = self._compute_log_proposal_prob_ratio(
            noise=noise[feasibility],
            scaled_noise=scaled_noise[feasibility],
            potential_curr=val_potential[feasibility],
            potential_prop=val_potential_prop[feasibility],
            precon_grad_potential_curr=precon_grad_potential[feasibility],
            precon_grad_potential_prop=precon_grad_potential_prop[feasibility],
            chol_cov_curr=chol_cov[feasibility],
            chol_cov_prop=chol_cov_prop[feasibility],
            stepsize=stepsize,
        )
        # reject by default
        reject = torch.ones(self.num_samples, dtype=torch.bool)
        log_uniform_vals = torch.empty_like(log_alpha_feas).uniform_().log_()

        # amongst those that are feasible, accept only if
        # log(U) <= log(alpha)
        reject[feasibility] = log_uniform_vals > log_alpha_feas

        # reject: size n
        # restore old particles, data
        # this restoration could be avoided if
        # we clone, but torch.clone is more expensive that torch.empty
        # and if the acceptance rate is reasonable, then torch.clone would be
        # more expensive that torch.empty + reject
        prop_particles[reject] = particles[reject]
        val_potential_prop[reject] = val_potential[reject]
        precon_grad_potential_prop[reject] = precon_grad_potential[reject]
        chol_cov_prop[reject] = chol_cov[reject]
        return (
            prop_particles,
            val_potential_prop,
            precon_grad_potential_prop,
            chol_cov_prop,
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
        particles_tracker = [particles]
        reject_tracker = []

        chol_covariance = self.CHOL(self.barrier.hessian(particles))
        # if diag_hess, n x d, else n x d x d
        val_potential, grad_potential = self.potential.value_and_gradient(particles)
        if self.barrier.diag_hess:
            precon_grad_potential = grad_potential / chol_covariance**2
        else:
            precon_grad_potential = torch.cholesky_solve(
                grad_potential.unsqueeze(-1), chol_covariance
            ).squeeze(-1)
        for _ in tqdm(range(num_iters), disable=no_progress):
            (
                particles,
                val_potential,
                precon_grad_potential,
                chol_covariance,
                reject,
            ) = self._update_particles(
                particles=particles,
                val_potential=val_potential,
                precon_grad_potential=precon_grad_potential,
                chol_cov=chol_covariance,
                stepsize=stepsize,
            )
            particles_tracker.append(particles)
            reject_tracker.append(reject)
        reject_tracker = torch.stack(reject_tracker)
        if return_particles:
            return torch.stack(particles_tracker), reject_tracker
        else:  # return final sample
            return particles, reject_tracker


class GeneralDikinSampler:
    def __init__(
        self,
        barrier: Barrier,
        potential: Optional[Potential],
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
        chol_cov_curr: torch.Tensor,
        chol_cov_prop: torch.Tensor,
    ):
        if self.potential is None:
            potential_diff = 0.0
        else:
            # all this computation assumes feasibility of iterates
            potential_diff = self.potential.value(particles) - self.potential.value(
                prop_particles
            )

        # scaled_noise is Sqrt[inv(cov)] @ noise
        if self.barrier.diag_hess:
            # when hessian is diagonal
            logdet_term = torch.log(chol_cov_prop).sum(dim=-1) - torch.log(
                chol_cov_curr
            ).sum(dim=-1)
            # tmp_quantity_1 is Lz^{T} @ Lx^{-T} @ \xi = Lz^{T} @ s\xi
            # Lz is n x d
            tmp_quantity_1 = chol_cov_prop * scaled_noise
        else:
            # when hessian is not diagonal
            # some computation is necessary
            # logdet is just sum of logs of the diagonals
            logdet_term = torch.diagonal(chol_cov_prop, dim1=-2, dim2=-1).log().sum(
                dim=-1
            ) - torch.diagonal(chol_cov_curr, dim1=-2, dim2=-1).log().sum(dim=-1)
            # tmp_quantity_1 is Lz^{T} @ Lx^{-T} @ \xi = Lz^{T} @ s\xi
            # Lz is n x d x d, and triangular
            tmp_quantity_1 = torch.einsum("bij,bi->bj", chol_cov_prop, scaled_noise)

        q1 = 0.5 * (
            torch.sum(torch.square(noise), dim=-1)
            - torch.sum(torch.square(tmp_quantity_1), dim=-1)
        )
        return q1 + potential_diff + logdet_term

    def _update_particles(
        self,
        particles: torch.Tensor,
        chol_cov: torch.Tensor,
        stepsize: float,
    ):
        # proposal step
        noise = torch.randn_like(particles)
        if self.barrier.diag_hess:
            # scaled_noise s\xi : Lx^{-1} * \xi
            # simpler computation
            scaled_noise = noise / chol_cov
        else:
            # scaled_noise s\xi : Lx^{-T} @ \xi
            # mat-vec
            scaled_noise = torch.linalg.solve_triangular(
                chol_cov.transpose(1, 2), noise.unsqueeze(-1), upper=True
            ).squeeze(-1)

        prop_particles = particles + math.sqrt(2 * stepsize) * scaled_noise
        feasibility = self.barrier.feasibility(prop_particles)
        chol_cov_prop = torch.empty_like(chol_cov)
        chol_cov_prop[feasibility] = self.CHOL(
            self.barrier.hessian(prop_particles[feasibility])
        )
        # chol_cov_prop: n x d if diagonal, n x d x d if not diagonal
        # full_cov_prop: n x d if diagonal, n x d x d if not diagonal

        # compute acceptance prob
        log_alpha_feas = self._compute_log_proposal_prob_ratio(
            noise=noise[feasibility],
            scaled_noise=scaled_noise[feasibility],
            particles=particles[feasibility],
            prop_particles=prop_particles[feasibility],
            chol_cov_curr=chol_cov[feasibility],
            chol_cov_prop=chol_cov_prop[feasibility],
        )
        # reject by default
        reject = torch.ones(self.num_samples, dtype=torch.bool)
        uniform_vals = torch.empty_like(log_alpha_feas).uniform_()
        reject[feasibility] = torch.log(uniform_vals) > log_alpha_feas
        # reject: n
        # restore old particles
        prop_particles[reject] = particles[reject]
        chol_cov_prop[reject] = chol_cov[reject]
        return (
            prop_particles,
            chol_cov_prop,
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
        particles_tracker = [particles]
        reject_tracker = []

        chol_covariance = self.CHOL(self.barrier.hessian(particles))
        # if diag_hess, n x d, else n x d x d
        for _ in tqdm(range(num_iters), disable=no_progress):
            (
                particles,
                chol_covariance,
                reject,
            ) = self._update_particles(
                particles=particles,
                chol_cov=chol_covariance,
                stepsize=stepsize,
            )
            reject_tracker.append(reject)
            if return_particles:
                particles_tracker.append(particles)
        reject_tracker = torch.stack(reject_tracker)
        if return_particles:
            return torch.stack(particles_tracker), reject_tracker
        else:  # return final sample
            return particles, reject_tracker
