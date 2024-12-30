import torch

torch.set_default_dtype(torch.float64)

from typing import List, Tuple

__all__ = [
    "Potential",
    "BayesianLogisticRegressionPotential",
    "DirichletPotential",
    "LinearPotential",
    "SumPotential",
]


class Potential:
    """
    Base class for Potentials
    """

    def __init__(self, *args, **kwargs):
        pass

    def feasibility(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def value(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def value_and_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.value(x), self.gradient(x)


class SumPotential(Potential):
    """
    Functionality to sum a list of potentials
    """

    def __init__(
        self,
        potentials: List[Potential],
    ):
        self.potentials = potentials

    def value(self, x: torch.Tensor):
        value = torch.zeros(x.shape[:-1])
        for potential in self.potentials:
            value.add_(potential.value(x))
        return value

    def gradient(self, x: torch.Tensor):
        gradient = torch.zeros_like(x)
        for potential in self.potentials:
            gradient.add_(potential.gradient(x))
        return gradient

    def value_and_gradient(self, x: torch.Tensor):
        value = torch.zeros(x.shape[:-1])
        gradient = torch.zeros_like(x)
        for potential in self.potentials:
            pval, pgrad = potential.value_and_gradient(x)
            value.add_(pval)
            gradient.add_(pgrad)
        return value, gradient


class DirichletPotential(Potential):
    """
    Dirichlet Potential
    """

    def __init__(self, alpha: torch.Tensor):
        self.dimension = alpha.shape[0] - 1  # the alpha is of length d + 1
        self._main_alpha = alpha[:-1]
        self._last_alpha = alpha[-1]

    @property
    def alpha(self):
        return torch.cat([self._main_alpha, self._last_alpha], dim=-1)

    def _safe_interior(self, x: torch.Tensor, squeeze_last_dim: bool):
        return torch.clamp_min(
            1 - torch.sum(x, dim=-1, keepdim=not squeeze_last_dim),
            min=1e-08,
        )

    def value(self, x: torch.Tensor):
        return (
            -torch.sum(self._main_alpha * torch.log(x), dim=-1)
            - torch.log(self._safe_interior(x=x, squeeze_last_dim=True))
            * self._last_alpha
        )

    def gradient(self, x: torch.Tensor):
        return -self._main_alpha / x + self._last_alpha / self._safe_interior(
            x=x, squeeze_last_dim=False
        )

    def value_and_gradient(self, x: torch.Tensor):
        slack = self._safe_interior(x, squeeze_last_dim=False)
        val = (
            -torch.sum(self._main_alpha * torch.log(x), dim=-1)
            - torch.log(slack.squeeze(dim=-1)) * self._last_alpha
        )
        grad = -self._main_alpha / x + self._last_alpha / slack
        return val, grad


class BayesianLogisticRegressionPotential(Potential):
    """
    Bayesian Logistic Regression Potential
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X  # shape N x d
        self.y = y  # shape N
        self.dimension = X.shape[1]

    def _inner_prod_X(self, thetas: torch.Tensor):
        return torch.einsum("...i,ni->...n", thetas, self.X)

    def value(self, thetas: torch.Tensor):
        inner_prod = self._inner_prod_X(thetas)  # ... x N
        t1 = torch.sum(torch.nn.functional.softplus(inner_prod), dim=-1)
        t2 = torch.einsum("...n,n->...", inner_prod, self.y)
        return t1 - t2

    def gradient(self, thetas: torch.Tensor):
        inner_prod = self._inner_prod_X(thetas)  # ... x N
        t1 = torch.einsum("...n,ni->...i", torch.sigmoid(inner_prod), self.X)  # ... x d
        t2 = self.X.transpose(0, 1) @ self.y  # d
        return t1 - t2

    def value_and_gradient(self, thetas: torch.Tensor):
        inner_prod = self._inner_prod_X(thetas)  # ... x N
        exp_neg_inner_prod = torch.exp(-inner_prod)  # ... x N
        tval1 = torch.sum(inner_prod + torch.log1p(exp_neg_inner_prod), dim=-1)  # ...
        tval2 = torch.einsum("...n,n->...", inner_prod, self.y)  # ...
        tgrad1 = torch.einsum(
            "...n,ni->...i", torch.reciprocal(1 + exp_neg_inner_prod), self.X
        )  # ... x d
        tgrad2 = self.X.transpose(0, 1) @ self.y  # d
        return tval1 - tval2, tgrad1 - tgrad2


class LinearPotential(Potential):
    """
    Linear potential
    """

    def __init__(self, sigma: torch.Tensor):
        self.sigma = sigma

    def value(self, x: torch.Tensor):
        return torch.sum(x * self.sigma, dim=-1)

    def gradient(self, x: torch.Tensor):
        return self.sigma.expand_as(x)


class QuadraticPotential(Potential):
    """
    Quadratic potential
    """

    def __init__(self, Q: torch.Tensor, r: torch.Tensor):
        # potential of the form
        # 0.5 * <x, Qx> - <r, x>
        self.Q = Q
        self.r = r

    def value(self, x: torch.Tensor):
        xQx = torch.einsum("...j,...i,ij->...", x, x, self.Q)
        rx = torch.sum(self.r * x, dim=-1)
        return 0.5 * xQx - rx

    def gradient(self, x: torch.Tensor):
        return torch.einsum("...j,ij->...i", x, self.Q) - self.r

    def value_and_gradient(self, x):
        Qx = torch.einsum("...j,ij->...i", x, self.Q)
        val = 0.5 * torch.sum(x * Qx, dim=-1) - torch.sum(self.r * x, dim=-1)
        grad = Qx - self.r
        return val, grad
