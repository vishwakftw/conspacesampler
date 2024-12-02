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


if __name__ == "__main__":
    unittest.main()
