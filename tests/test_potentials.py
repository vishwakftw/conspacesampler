import torch

torch.set_default_dtype(torch.float64)

import unittest

from conspacesampler.potentials import (
    DirichletPotential,
    BayesianLogisticRegressionPotential,
    SumPotential,
)


class TestDirichletPotential(unittest.TestCase):
    def test_value(self):
        dimensions = [3, 5, 7, 11]
        alphas = [0.2, 0.5, 1.0, 2.0, 5.0]
        for dim in dimensions:
            for alpha in alphas:
                full_alpha = torch.ones(dim + 1) * alpha
                torch_dirichlet = torch.distributions.Dirichlet(
                    concentration=full_alpha + 1
                )
                my_dirichlet = DirichletPotential(alpha=full_alpha)
                samples = torch_dirichlet.sample((17,))
                torch_val = torch_dirichlet.log_prob(samples)
                my_val = -my_dirichlet.value(
                    samples[:, :-1]
                ) - torch_dirichlet._log_normalizer(full_alpha + 1)
                self.assertTrue(
                    torch.allclose(torch_val, my_val), "log_prob doesn't agree"
                )

    def test_gradient(self):
        dimensions = [3, 5, 7, 11]
        alphas = [0.2, 0.5, 1.0, 2.0, 5.0]
        for dim in dimensions:
            for alpha in alphas:
                full_alpha = torch.ones(dim + 1) * alpha
                torch_dirichlet = torch.distributions.Dirichlet(
                    concentration=full_alpha + 1
                )
                my_dirichlet = DirichletPotential(alpha=full_alpha)
                samples = torch_dirichlet.sample((17,))[:, :-1].clone().requires_grad_()

                vals = my_dirichlet.value(samples).sum()
                vals.backward()
                with torch.no_grad():
                    self.assertTrue(
                        torch.allclose(samples.grad, my_dirichlet.gradient(samples)),
                        "gradient and autograd gradient don't agree",
                    )

    def test_value_and_gradient(self):
        dimensions = [3, 5, 7, 11]
        alphas = [0.2, 0.5, 1.0, 2.0, 5.0]
        for dim in dimensions:
            for alpha in alphas:
                full_alpha = torch.ones(dim + 1) * alpha
                torch_dirichlet = torch.distributions.Dirichlet(
                    concentration=full_alpha + 1
                )
                my_dirichlet = DirichletPotential(alpha=full_alpha)
                samples = torch_dirichlet.sample((17,))[:, :-1]
                value = my_dirichlet.value(samples)
                gradient = my_dirichlet.gradient(samples)
                val_and_grad = my_dirichlet.value_and_gradient(samples)
                self.assertTrue(
                    torch.allclose(value, val_and_grad[0]),
                    "value and value_and_gradient[0] don't agree",
                )

                self.assertTrue(
                    torch.allclose(gradient, val_and_grad[1]),
                    "gradient and value_and_gradient[1] don't agree",
                )


class TestBayesianLogisticRegressionPotential(unittest.TestCase):
    def setUp(self):
        self.dim = 17
        self.N = 37
        self.X = torch.rand(self.N, self.dim) * 5 - 2.5
        self.y = torch.empty(self.N).bernoulli_()
        self.blr_potential = BayesianLogisticRegressionPotential(self.X, self.y)

    def test_value(self):
        for _ in range(19):
            thetas = torch.randn(23, self.dim)
            my_potential = self.blr_potential.value(thetas=thetas)
            logits = torch.einsum("bj,ij->bi", thetas, self.X)
            for i in range(23):
                torch_potential = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[i], self.y, reduction="sum"
                )
                self.assertTrue(
                    torch.allclose(my_potential[i], torch_potential),
                    "potential and cross-entropy loss don't agree",
                )

    def test_gradient(self):
        for _ in range(19):
            thetas = torch.randn(23, self.dim).requires_grad_()
            vals = self.blr_potential.value(thetas=thetas).sum()
            vals.backward()
            with torch.no_grad():
                self.assertTrue(
                    torch.allclose(
                        self.blr_potential.gradient(thetas=thetas), thetas.grad
                    ),
                    "gradient and autograd gradient don't agree",
                )

    def test_value_and_gradient(self):
        for _ in range(19):
            thetas = torch.randn(23, self.dim)
            value = self.blr_potential.value(thetas=thetas)
            gradient = self.blr_potential.gradient(thetas=thetas)
            val_and_grad = self.blr_potential.value_and_gradient(thetas=thetas)
            self.assertTrue(
                torch.allclose(value, val_and_grad[0]),
                "value and value_and_gradient[0] don't agree",
            )

            self.assertTrue(
                torch.allclose(gradient, val_and_grad[1]),
                "gradient and value_and_gradient[1] don't agree",
            )


class TestSumPotential(unittest.TestCase):
    def setUp(self):
        self.dim = 29
        self.N = 37
        self.X = torch.rand(self.N, self.dim) * 5 - 2.5
        self.y = torch.empty(self.N).bernoulli_()
        self.potentials = [
            DirichletPotential(alpha=torch.ones(self.dim + 1)),
            BayesianLogisticRegressionPotential(X=self.X, y=self.y),
        ]

    def _test_attr(self, attr: str):
        for _ in range(19):
            thetas = torch.rand(23, self.dim + 1)
            thetas /= thetas.sum(dim=-1, keepdim=True)
            thetas = thetas[:, :-1]
            my_sum_potential = SumPotential(self.potentials)
            my_sum_val = getattr(my_sum_potential, attr)(thetas)
            my_comp_vals = torch.stack(
                [getattr(pot, attr)(thetas) for pot in self.potentials], dim=-1
            ).sum(dim=-1)
            self.assertTrue(
                torch.allclose(
                    my_sum_val,
                    my_comp_vals,
                ),
                "stack and sum doesn't match sum of values from SumPotential",
            )

    def test_value(self):
        self._test_attr(attr="value")

    def test_gradient(self):
        self._test_attr(attr="gradient")


if __name__ == "__main__":
    unittest.main()
