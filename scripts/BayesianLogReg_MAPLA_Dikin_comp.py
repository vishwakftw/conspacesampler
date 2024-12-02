import torch

torch.set_default_dtype(torch.float64)

from argparse import ArgumentParser
from time import time

from conspacesampler.algorithms.natural_algorithms import (
    GeneralMAPLASampler,
    GeneralDikinSampler,
)
from conspacesampler.barriers import PolytopeBarrier
from conspacesampler.potentials import BayesianLogisticRegressionPotential
from conspacesampler.utils import define_polytope


def run_sampling(
    domain_index: int,
    dimension: int,
    num_iters: int,
    stepsize_scale: float,
    run_index: int,
    num_particles: int,
    progress_file: str,
    error_progress_file: str,
    error_quantile_file: str,
    show_tqdm: bool = False,
):
    random_seed = int(dimension * 3141 + domain_index)
    torch.manual_seed(random_seed)

    # the above seeding ensures consistency across different run_index
    # the randomness in the run_index arises from the datasets

    theta_star = torch.ones(dimension)
    translation = torch.ones(dimension) * 0.5
    polytope = define_polytope(
        dimension=dimension,
        bounds=2 * torch.ones(dimension),
        num_rotations=dimension // 2,
        translation=translation,
    )
    barrier = PolytopeBarrier(polytope=polytope)
    assert barrier.feasibility(theta_star).all(), "theta_star not feasible"

    # create a random seed based on the values
    random_seed = int(dimension * 3141 + run_index)
    # same random_seed for a specification of stepsize
    # this is to better judge the effects of these on the same sampling problem.
    # same random_seed for a specification of num_particles as well.

    # reseed the generator
    torch.manual_seed(random_seed)

    num_datapoints = 20 * dimension

    X = torch.empty(num_datapoints, dimension).bernoulli_() * 2 - 1
    X /= torch.linalg.norm(X, dim=-1, keepdim=True)
    y = (torch.rand(num_datapoints) < torch.sigmoid(X @ theta_star)).to(dtype=X.dtype)
    lmax = torch.linalg.eigvalsh(X.T @ X).max()

    stepsize = stepsize_scale / (dimension * lmax)

    potential = BayesianLogisticRegressionPotential(X=X, y=y)
    initial_particles = torch.rand(num_particles, dimension) * 0.2 - 0.1
    assert barrier.feasibility(initial_particles).all(), "all initial not feasible"

    sampler_mapla = GeneralMAPLASampler(
        barrier=barrier, potential=potential, num_samples=num_particles
    )
    sampler_dikin = GeneralDikinSampler(
        barrier=barrier, potential=potential, num_samples=num_particles
    )
    sampler_mapla.set_initial_particles(initial_particles)
    sampler_dikin.set_initial_particles(initial_particles)

    time_start = time()
    particles_mapla, rejection_masks = sampler_mapla.mix(
        num_iters=num_iters,
        stepsize=stepsize,
        no_progress=not show_tqdm,
        return_particles=True,
    )
    time_end = time()
    total_time_mapla = time_end - time_start
    acceptance_full = num_particles - torch.sum(rejection_masks, dim=-1)

    # compute avg_acceptance after the first tenth of iterations
    avg_acceptance_mapla = torch.mean(
        acceptance_full[int(0.1 * num_iters) :] / num_particles
    )
    error_mapla = (
        torch.sum(torch.abs(torch.mean(particles_mapla, dim=-2) - theta_star), dim=-1)
        / dimension
    )
    nll_mapla = potential.value(torch.mean(particles_mapla, dim=-2)) / num_datapoints

    time_start = time()
    particles_dikin, rejection_masks = sampler_dikin.mix(
        num_iters=num_iters,
        stepsize=stepsize,
        no_progress=not show_tqdm,
        return_particles=True,
    )
    time_end = time()
    total_time_dikin = time_end - time_start
    acceptance_full = num_particles - torch.sum(rejection_masks, dim=-1)

    # compute avg_acceptance after the first tenth of iterations
    avg_acceptance_dikin = torch.mean(
        acceptance_full[int(0.1 * num_iters) :] / num_particles
    )
    error_dikin = (
        torch.sum(torch.abs(torch.mean(particles_dikin, dim=-2) - theta_star), dim=-1)
        / dimension
    )
    nll_dikin = potential.value(torch.mean(particles_dikin, dim=-2)) / num_datapoints

    for alg, errors, nlls, total_time, avg_acceptance in zip(
        ["MAPLA", "Dikin"],
        [error_mapla, error_dikin],
        [nll_mapla, nll_dikin],
        [total_time_mapla, total_time_dikin],
        [avg_acceptance_mapla.item(), avg_acceptance_dikin.item()],
    ):
        for itr, (error, nll) in enumerate(zip(errors, nlls)):
            if error_progress_file == "NA":
                continue

            if (itr % 5 != 0) and (itr != errors.shape[0] - 1):
                continue

            desc_string = (
                f"{alg},{dimension},{stepsize_scale},{domain_index},"
                f"{run_index},{num_particles},{itr},{error:.6f},{nll:.6f}"
            )
            with open(error_progress_file, "a") as f:
                f.write(desc_string + "\n")

        desc_string = (
            f"{alg},{dimension},{num_iters},"
            f"{stepsize_scale},{domain_index},{run_index},{num_particles},"
            f"{total_time:.4f},{avg_acceptance:.6f},{error:.6f},{nll:.6f}"
        )

        with open(progress_file, "a") as f:
            f.write(desc_string + "\n")

    if error_quantile_file != "NA":
        for alg, particles in zip(
            ["MAPLA", "Dikin"], [particles_mapla, particles_dikin]
        ):
            avg_error = torch.mean(particles - theta_star, dim=-1)
            error_quantiles = torch.quantile(
                avg_error, q=torch.tensor([0.25, 0.50, 0.75]), dim=-1
            )

            for itr, (q25, q50, q75) in enumerate(
                zip(error_quantiles[0], error_quantiles[1], error_quantiles[2])
            ):

                if (itr % 5 != 0) and (itr != errors.shape[0] - 1):
                    continue

                desc_string = (
                    f"{alg},{dimension},{stepsize_scale},{domain_index},"
                    f"{run_index},{num_particles},{itr},"
                    f"{q25:.6f},{q50:.6f},{q75:.6f}"
                )
                with open(error_quantile_file, "a") as f:
                    f.write(desc_string + "\n")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--dimension", type=int, help="Dimension")
    p.add_argument("--num_iters", type=int, help="Number of iterations")
    p.add_argument("--stepsize_scale", type=float, help="Step size scaling parameter")
    p.add_argument(
        "--domain_index",
        type=int,
        default=153,
        help="Domain index (for polytope generation)",
    )
    p.add_argument("--run_index", type=int, help="Run index for multiple runs")
    p.add_argument("--num_particles", type=int, help="Number of samples")
    p.add_argument(
        "--show_tqdm", action="store_true", help="Toggle for showing progress bar"
    )
    p.add_argument(
        "--progress_file",
        default="progress.txt",
        type=str,
        help="Location to store results",
    )
    p.add_argument(
        "--error_progress_file",
        default="NA",
        type=str,
        help="Location to store errors",
    )
    p.add_argument(
        "--error_quantile_file",
        default="NA",
        type=str,
        help="Perform error coverage analysis",
    )
    args = p.parse_args()
    args = vars(args)
    run_sampling(**args)
