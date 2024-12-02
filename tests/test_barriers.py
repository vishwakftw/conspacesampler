import torch

torch.set_default_dtype(torch.float64)

import unittest

from conspacesampler.barriers import (
    BoxBarrier,
    EllipsoidBarrier,
    SimplexBarrier,
    ComposeBarrier,
)
from conspacesampler.utils import define_ellipsoid


class TestBoxBarrier(unittest.TestCase):
    # Let's test the implementation
    def _test_feasibility_single(self, box_barrier: BoxBarrier):
        x = box_barrier.sample_uniform(2)[0]
        self.assertTrue(
            torch.all(box_barrier.feasibility(x)), msg=str(box_barrier.feasibility(x))
        )

        y = torch.tile(torch.Tensor([-1, 1]), (box_barrier.dimension, 1)).T
        y = y * (box_barrier.bounds + 1)  # outside
        self.assertFalse(
            torch.any(box_barrier.feasibility(y)),
            msg=str(box_barrier.feasibility(y)),
        )

    def test_feasibility(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            bounds = torch.rand(dim) * 4 + 0.1  # bounds can take values in (0.1, 4.1)
            box_barrier = BoxBarrier(bounds=bounds)
            self._test_feasibility_single(box_barrier)

    def _test_gradient_inverse_gradient_single(self, box_barrier: BoxBarrier):
        for i in range(13):
            x = box_barrier.sample_uniform(2)[0]
            y = torch.randn(2, box_barrier.dimension)

            grad = box_barrier.gradient(x)
            inv_grad = box_barrier.inverse_gradient(y)

            self.assertTrue(
                torch.allclose(
                    box_barrier.inverse_gradient(grad), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(torch.max(torch.abs(box_barrier.inverse_gradient(grad) - x))),
            )
            self.assertTrue(
                torch.allclose(
                    box_barrier.gradient(inv_grad), y, rtol=1e-04, atol=1e-06
                ),
                msg=str(torch.max(torch.abs(box_barrier.gradient(inv_grad) - y))),
            )

    def test_gradient_inverse_gradient(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            bounds = torch.rand(dim) * 4 + 0.1
            box_barrier = BoxBarrier(bounds=bounds)
            self._test_gradient_inverse_gradient_single(box_barrier)


class TestEllipsoidBarrier(unittest.TestCase):
    # Let's test the implementation
    def _test_feasibility_single(self, ellipsoid_barrier: EllipsoidBarrier):
        x = torch.randn(2, ellipsoid_barrier.dimension)
        x = (
            x
            / torch.sqrt(
                ellipsoid_barrier._ellipsoid_inner_product(x).unsqueeze(dim=-1)
            )
            * torch.rand(1)
        )  # to be in the interior
        self.assertTrue(
            torch.all(ellipsoid_barrier.feasibility(x)),
            msg=str(ellipsoid_barrier.feasibility(x)),
        )

        y = torch.randn(2, ellipsoid_barrier.dimension)
        y = (
            y
            / torch.sqrt(
                ellipsoid_barrier._ellipsoid_inner_product(x).unsqueeze(dim=-1)
            )
            * (torch.rand(1) + 1)
        )  # to be in the exterior
        self.assertFalse(
            torch.any(ellipsoid_barrier.feasibility(y)),
            msg=str(ellipsoid_barrier.feasibility(y)),
        )

    def test_feasibility(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            ellipsoid = define_ellipsoid(dim, random_seed=97)
            ellipsoid_barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
            self._test_feasibility_single(ellipsoid_barrier)

    def _test_ellipsoid_map_inverse_single(self, ellipsoid_barrier: EllipsoidBarrier):
        for i in range(13):
            x = torch.randn(2, ellipsoid_barrier.dimension)
            Ax = ellipsoid_barrier._ellipsoid_map(x)
            invAx = ellipsoid_barrier._inverse_ellipsoid_map(x)

            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier._inverse_ellipsoid_map(Ax),
                    x,
                    rtol=1e-04,
                    atol=1e-06,
                ),
                msg=str(
                    torch.max(
                        torch.abs(ellipsoid_barrier._inverse_ellipsoid_map(Ax) - x)
                    )
                ),
            )

            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier._ellipsoid_map(invAx), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(
                    torch.max(torch.abs(ellipsoid_barrier._ellipsoid_map(invAx) - x))
                ),
            )

    def test_ellipsoid_map_inverse(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            ellipsoid = define_ellipsoid(dim, random_seed=97)
            ellipsoid_barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
            self._test_ellipsoid_map_inverse_single(ellipsoid_barrier)

    def _test_gradient_inverse_gradient_single(
        self, ellipsoid_barrier: EllipsoidBarrier
    ):
        for i in range(13):
            x = torch.randn(2, ellipsoid_barrier.dimension)
            y = torch.randn(2, ellipsoid_barrier.dimension)
            x = (
                x
                / torch.sqrt(
                    ellipsoid_barrier._ellipsoid_inner_product(x).unsqueeze(dim=-1)
                )
                * torch.rand(1)
            )  # to be in the interior

            grad = ellipsoid_barrier.gradient(x)
            inv_grad = ellipsoid_barrier.inverse_gradient(y)

            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier.inverse_gradient(grad), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(
                    torch.max(torch.abs(ellipsoid_barrier.inverse_gradient(grad) - x))
                ),
            )
            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier.gradient(inv_grad),
                    y,
                    atol=1e-06,
                    rtol=1e-04,
                ),
                msg=str(torch.max(torch.abs(ellipsoid_barrier.gradient(inv_grad) - y))),
            )

    def test_gradient_inverse_gradient(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            ellipsoid = define_ellipsoid(dim, random_seed=97)
            ellipsoid_barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
            self._test_gradient_inverse_gradient_single(ellipsoid_barrier)


class TestSimplexBarrier(unittest.TestCase):
    # Let's test the implementation
    def _test_feasibility_single(self, simplex_barrier: SimplexBarrier):
        x = torch.rand(2, simplex_barrier.dimension)
        x = x / x.sum(dim=-1, keepdims=True) * torch.rand(2, 1)
        self.assertTrue(
            torch.all(simplex_barrier.feasibility(x)),
            msg=str(simplex_barrier.feasibility(x)),
        )

        y = torch.rand(2, simplex_barrier.dimension)
        y = y / y.sum(dim=-1, keepdims=True) * (torch.rand(2, 1) + 1.1)
        self.assertFalse(
            torch.any(simplex_barrier.feasibility(y)),
            msg=str(simplex_barrier.feasibility(y)),
        )

    def test_feasibility(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            simplex_barrier = SimplexBarrier(dimension=dim)
            self._test_feasibility_single(simplex_barrier)

    def _test_gradient_inverse_gradient_single(self, simplex_barrier: SimplexBarrier):
        for i in range(13):
            x = torch.rand(2, simplex_barrier.dimension)
            y = torch.randn(2, simplex_barrier.dimension)
            x = (
                x / x.sum(dim=-1, keepdims=True) * torch.Tensor([[0.5], [0.99]])
            )  # to be in the interior

            # two cases:
            # points close to the boundary
            # points far away from the boundary (well in the interior)

            grad = simplex_barrier.gradient(x)
            inv_grad = simplex_barrier.inverse_gradient(y)

            self.assertTrue(
                torch.allclose(
                    simplex_barrier.inverse_gradient(grad), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(
                    torch.max(torch.abs(simplex_barrier.inverse_gradient(grad) - x))
                ),
            )
            self.assertTrue(
                torch.allclose(
                    simplex_barrier.gradient(inv_grad), y, rtol=1e-04, atol=1e-06
                ),
                msg=str(torch.max(torch.abs(simplex_barrier.gradient(inv_grad) - y))),
            )

    def test_gradient_inverse_gradient(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            simplex_barrier = SimplexBarrier(dimension=dim)
            self._test_gradient_inverse_gradient_single(simplex_barrier)


class TestComposeBarrier(unittest.TestCase):
    def setUp(self):
        self.dim = 43
        self.barriers = [
            BoxBarrier(bounds=torch.ones(self.dim)),
            EllipsoidBarrier(
                ellipsoid={"eigvals": torch.ones(self.dim), "rot": torch.eye(self.dim)}
            ),
        ]

    def test_feasibility(self):
        # note that the ball is inside the box
        x = torch.randn(19, self.dim)
        x = (
            x
            / torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
            * torch.rand(1)
        )
        my_composed_barrier = ComposeBarrier(self.barriers)
        self.assertTrue(
            my_composed_barrier.feasibility(x).all(), "invalid check for feasibility"
        )

        # consider points outside the ball but inside the box
        x = torch.empty(self.dim).bernoulli_() * 2 - 1
        x = x * torch.rand(19, 1) * (self.dim**0.5 - 1) + 1
        self.assertFalse(
            my_composed_barrier.feasibility(x).all(), "invalid check for feasibility"
        )

    def test_hessian(self):
        x = torch.randn(19, self.dim)
        x = (
            x
            / torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
            * torch.rand(1)
        )
        my_composed_barrier = ComposeBarrier(self.barriers)
        my_hessian = my_composed_barrier.hessian(x=x)
        manual_hessian = self.barriers[0].hessian(x=x).diag_embed()
        manual_hessian += self.barriers[1].hessian(x=x)
        self.assertTrue(
            torch.allclose(
                my_hessian,
                manual_hessian,
            ),
            "invalid composed Hessian",
        )


if __name__ == "__main__":
    unittest.main()
