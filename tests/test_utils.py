import scipy.stats as sts
import torch

torch.set_default_dtype(torch.float64)

import unittest

from conspacesampler.barriers import BoxBarrier, EllipsoidBarrier, PolytopeBarrier
from conspacesampler.utils import (
    compute_bounds_box,
    compute_bounds_polytope,
    compute_bounds_ellipsoid,
    define_box,
    define_ellipsoid,
    energy_distance,
    kstest_statistic,
)


class TestComputeBounds(unittest.TestCase):
    def test_equivalence_box_polytope(self):
        dimensions = [3, 5, 7, 11]
        consts = [1.0, 2.0, 5.0, 10.0]

        for dim in dimensions:
            for c in consts:
                bounds = define_box(dimension=dim, condition_number=c)
                boxbarrier = BoxBarrier(bounds=bounds)
                polyA = torch.cat([torch.eye(dim), -torch.eye(dim)], dim=0)
                polyb = torch.cat([bounds, bounds], dim=0)
                polytopebarrier = PolytopeBarrier({"A": polyA, "b": polyb})
                x = torch.rand(23, dim) * (2 * bounds) - bounds
                u = torch.randn(23, dim)
                u /= torch.linalg.norm(u, dim=-1, keepdim=True)
                rlowb, rhighb = compute_bounds_box(
                    barrier=boxbarrier, particles=x, directions=u
                )
                rlowp, rhighp = compute_bounds_polytope(
                    barrier=polytopebarrier, particles=x, directions=u
                )
                self.assertTrue(
                    torch.allclose(rlowb, rlowp),
                    "Polytope lower bounds and box lower bounds don't agree: "
                    f"max error: {torch.max(torch.abs(rlowb - rlowp))}",
                )
                self.assertTrue(
                    torch.allclose(rhighb, rhighp),
                    "Polytope lower bounds and box upper bounds don't agree: "
                    f"max error: {torch.max(torch.abs(rhighb - rhighp))}",
                )

                self.assertTrue(
                    torch.all(boxbarrier.feasibility(x + rlowb * u)),
                    "Infeasible bounds",
                )
                self.assertTrue(
                    torch.all(boxbarrier.feasibility(x + rhighb * u)),
                    "Infeasible bounds",
                )

    def test_ellipsoid(self):
        dimensions = [3, 5, 7, 11]
        consts = [1.0, 2.0, 5.0, 10.0]

        for dim in dimensions:
            for c in consts:
                ellipsoid = define_ellipsoid(
                    dimension=dim, random_seed=dim, condition_number=c
                )
                ellipsoidbarrier = EllipsoidBarrier(ellipsoid=ellipsoid)
                x = torch.randn(23, dim)
                x = (
                    x
                    / torch.sqrt(
                        ellipsoidbarrier._ellipsoid_inner_product(x).unsqueeze(dim=-1)
                    )
                    * torch.rand(1)
                )  # to be in the interior
                u = torch.randn(23, dim)
                u /= torch.linalg.norm(u, dim=-1, keepdim=True)
                rlowe, rhighe = compute_bounds_ellipsoid(
                    barrier=ellipsoidbarrier, particles=x, directions=u
                )
                self.assertTrue(
                    torch.all(ellipsoidbarrier.feasibility(x + rlowe * u)),
                    "Infeasible bounds",
                )
                self.assertTrue(
                    torch.all(ellipsoidbarrier.feasibility(x + rhighe * u)),
                    "Infeasible bounds",
                )


class TestKSTestStatistic(unittest.TestCase):
    def test_gaussian(self):
        dimensions = [5, 7, 11]

        for dim in dimensions:
            mu, sigma = torch.rand(dim) * 4 - 2, torch.rand(dim) * 1.5 + 0.5
            vals = torch.randn(3, 19, dim) * sigma + mu
            # first batch dimension is NOT gaussian
            # second and third batch dimensions are gaussian
            vals[0] = torch.rand(19, dim) * (2 * sigma) + (mu - sigma)
            vals = vals.sort(dim=-2).values
            cdf_vals = torch.distributions.Normal(loc=mu, scale=sigma).cdf(vals)
            actual = kstest_statistic(cdf_vals=cdf_vals, reduce_max=False)
            for b in range(3):
                for i, (mu_i, sigma_i) in enumerate(zip(mu.numpy(), sigma.numpy())):
                    exp_i = sts.ks_1samp(
                        vals[b, :, i].numpy(), cdf=sts.norm.cdf, args=(mu_i, sigma_i)
                    )
                    self.assertTrue(
                        torch.allclose(actual[b, i], torch.tensor(exp_i.statistic))
                    )

    def test_gamma(self):
        dimensions = [5, 7, 11]

        for dim in dimensions:
            conc, rate = torch.rand(dim) * 3, torch.rand(dim) * 4
            gamma_dist = torch.distributions.Gamma(concentration=conc, rate=rate)
            vals = gamma_dist.sample((3, 19))
            # first batch dimension is NOT gamma
            # second and third batch dimensions are gamma
            vals[0] = torch.rand(19, dim) * 2 * (conc / rate)
            vals = vals.sort(dim=-2).values
            cdf_vals = gamma_dist.cdf(vals)
            actual = kstest_statistic(cdf_vals=cdf_vals, reduce_max=False)
            for b in range(3):
                for i, (conc_i, rate_i) in enumerate(zip(conc.numpy(), rate.numpy())):
                    exp_i = sts.ks_1samp(
                        vals[b, :, i].numpy(),
                        cdf=sts.gamma.cdf,
                        args=(conc_i, 0, 1 / rate_i),
                    )
                    self.assertTrue(
                        torch.allclose(actual[b, i], torch.tensor(exp_i.statistic))
                    )


class TestEnergyDistance(unittest.TestCase):
    def test_edstat(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            x = torch.randn(17, 53, dim)

            # ed of x with itself is 0
            self.assertTrue(torch.allclose(energy_distance(x, x), torch.tensor(0.0)))

            # test batching behaviour
            y = torch.randn(17, 59, dim)
            ED = energy_distance(x, y)
            for b in range(17):
                self.assertTrue(torch.allclose(ED[b], energy_distance(x[b], y[b])))

            # test batching behaviour with broadcasting
            y = torch.randn(59, dim)
            ED = energy_distance(x, y)
            for b in range(17):
                self.assertTrue(torch.allclose(ED[b], energy_distance(x[b], y)))


if __name__ == "__main__":
    unittest.main()
