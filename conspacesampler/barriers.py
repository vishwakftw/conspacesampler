import torch

torch.set_default_dtype(torch.float64)

from typing import Dict, List

__all__ = [
    "Barrier",
    "BoxBarrier",
    "ComposeBarrier",
    "EllipsoidBarrier",
    "PolytopeBarrier",
    "SimplexBarrier",
]


class Barrier:
    """
    Base class for Barriers
    """

    diag_hess = False
    ZERO = torch.tensor(0.0)

    def __init__(self, *args, **kwargs):
        pass

    def feasibility(self, x: torch.Tensor):
        """
        Returns if x is feasible.
        """
        raise NotImplementedError

    def value(self, x: torch.Tensor):
        raise NotImplementedError

    def gradient(self, x: torch.Tensor):
        raise NotImplementedError

    def inverse_gradient(self, y: torch.Tensor):
        raise NotImplementedError

    def hessian(self, x: torch.Tensor):
        raise NotImplementedError

    def sample_uniform(self, n_points: int):
        raise NotImplementedError

    def boundary_to_interior_half(self, x: torch.Tensor):
        raise NotImplementedError


class ComposeBarrier(Barrier):
    """
    Functionality to compose a list of barriers
    """

    def __init__(
        self,
        barriers: List[Barrier],
    ):
        # barriers and any dimension that have resulted from lifting
        # for that barrier
        self.barriers = barriers
        _hess_compute_order = [
            [],
            [],
        ]  # first for diag_hess = False, second for diag_hess = True
        for i, barrier in enumerate(barriers):
            _hess_compute_order[int(barrier.diag_hess)].append(i)
        if len(_hess_compute_order[0]) == 0:
            self.diag_hess = True
        self._hess_compute_order = _hess_compute_order

    def feasibility(self, x: torch.Tensor):
        check = torch.ones(x.shape[:-1], dtype=torch.bool)
        for barrier in self.barriers:
            check.logical_and_(barrier.feasibility(x))
        return check

    def value(self, x: torch.Tensor):
        value = torch.zeros(x.shape[:-1])
        for barrier in self.barriers:
            value.add_(barrier.value(x))
        return value

    def gradient(self, x: torch.Tensor):
        gradient = torch.zeros_like(x)
        for barrier in self.barriers:
            gradient.add_(barrier.gradient(x))
        return gradient

    def hessian(self, x: torch.Tensor):
        if len(self._hess_compute_order[1]) > 0:
            hessian = torch.zeros_like(x)
            for idx in self._hess_compute_order[1]:  # do for diag_hess = True first
                hessian += self.barriers[idx].hessian(x)

        if not self.diag_hess:
            hessian = torch.diag_embed(hessian)
            for idx in self._hess_compute_order[0]:  # do for diag_hess = False next
                hessian += self.barriers[idx].hessian(x)

        # this way, if diag_hess is True, the embed is not performed
        # and if diag_hess is False, then the embed is performed just once
        return hessian


class BoxBarrier(Barrier):
    """
    Log barrier of box
    """

    def __init__(self, bounds: torch.Tensor):
        self.bounds = bounds  # [a1, a2, ...] form the box [-a1, a1] x [-a2, a2] x ...
        self.dimension = bounds.shape[-1]
        self.diag_hess = True

    def _safe_diff(self, x: torch.Tensor):
        """
        Returns bounds ** 2 - x ** 2, but with care around the boundary
        """
        return torch.clamp_min(torch.square(self.bounds) - torch.square(x), min=1e-08)

    def feasibility(self, x: torch.Tensor):
        """
        Returns if x is feasible.
        """
        return torch.all(torch.abs(x) <= self.bounds, dim=-1)

    def value(self, x: torch.Tensor):
        """
        Computes the value of the potential at x
        defined as -log(1 - <x, Ax>) * c
        where A is the ellipsoid and c is the inverse temperature
        """
        return -torch.sum(torch.log(self._safe_diff(x)), dim=-1)

    def gradient(self, x: torch.Tensor):
        """
        Computes the gradient of the potential at x
        defined as 2 * x_{i} / (a_{i} ** 2 - x_{i} ** 2)
        """
        return 2 * x / self._safe_diff(x)

    def inverse_gradient(self, y: torch.Tensor):
        """
        Computes the inverse of the gradient of the potential at y.
        Element-wise defined as (-1 + sqrt(1 + a[i]^2 y[i]^2)) / y[i].
        """
        asqr_ysqr = torch.square(self.bounds * y)
        raw = (-1 + torch.sqrt(1 + asqr_ysqr)) / y
        raw[torch.isclose(y, self.ZERO, atol=1e-07, rtol=1e-05)] = (
            0  # if very close to 0, then 0
        )
        return raw

    def hessian(self, x: torch.Tensor):
        """
        Computes the diagonal of the hessian (in this case is diagonal)
        """
        reci_diff = self._safe_diff(x).reciprocal_()
        output = 2 * reci_diff + 4 * torch.square(x * reci_diff)
        return output.clamp_max_(1e07)

    def sample_uniform(self, n_points: int):
        """
        Sampling points uniformly from a box
        """
        return torch.rand(n_points, self.dimension) * (2 * self.bounds) - self.bounds

    def boundary_to_interior_half(self, x: torch.Tensor):
        """
        Returns if x belongs in the region between the
        boundary and the interior where the volume of this
        region is 0.5 under the uniform distribution
        """
        return torch.any(
            torch.abs(x) > self.bounds * 0.5 ** (1 / self.dimension), dim=-1
        )


class EllipsoidBarrier(Barrier):
    """
    Log barrier for Ellipsoid
    """

    def __init__(self, ellipsoid: Dict[str, torch.Tensor]):
        self.ellipsoid = ellipsoid
        self.dimension = ellipsoid["eigvals"].shape[0]

    def _ellipsoid_inner_product(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes <x, Ax> efficiently.
        """
        UTx = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, x)
        return torch.sum(torch.square(UTx) * self.ellipsoid["eigvals"], dim=-1)

    def _inverse_ellipsoid_inner_product(self, y: torch.Tensor):
        """
        Computes <y, inv(A)y> efficiently.
        """
        UTy = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, y)
        return torch.sum(torch.square(UTy) / self.ellipsoid["eigvals"], dim=-1)

    def _ellipsoid_map(self, x: torch.Tensor):
        """
        Computes Ax efficiently.
        """
        UTx = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, x)
        return torch.einsum(
            "ij,...j->...i", self.ellipsoid["rot"] * self.ellipsoid["eigvals"], UTx
        )

    def _inverse_ellipsoid_map(self, y: torch.Tensor):
        """
        Computes inv(A)y efficiently.
        """
        UTx = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, y)
        return torch.einsum(
            "ij,...j->...i", self.ellipsoid["rot"] / self.ellipsoid["eigvals"], UTx
        )

    def feasibility(self, x: torch.Tensor):
        """
        Returns if x is feasible.
        """
        return self._ellipsoid_inner_product(x) <= 1

    def value(self, x: torch.Tensor):
        """
        Computes the value of the potential at x
        defined as -log(1 - <x, Ax>)
        where A is the ellipsoid
        """
        inner_product = self._ellipsoid_inner_product(x)
        inner_product[inner_product >= 1] = 0  # points outside or on the boundary
        return -torch.log1p(-inner_product)

    def gradient(self, x: torch.Tensor):
        """
        Computes the gradient of the potential at x
        defined as 2 * (Ax) / (1 - <x, Ax>)
        where A is the ellipsoid
        """
        one_minus_xAx = 1 - self._ellipsoid_inner_product(x)
        # NOTE: when x is too close to the boundary
        # this might be really small and cause problems for us when
        # dividing by 1 - <x, Ax>.
        one_minus_xAx.clamp_min_(min=1e-08)
        Ax = self._ellipsoid_map(x)
        return 2 * Ax / one_minus_xAx.unsqueeze(dim=-1)

    def inverse_gradient(self, y: torch.Tensor):
        """
        Computes the inverse of the potential at y
        defined as x which satisfies gradient(x) = y.
        This is also equal to the gradient of the Fenchel-Legendre dual.
        """
        # We did some complicated calculations for the inverse of the function
        # g(x) = x / (1 - <x, Ax>).
        # inv(g)(y) = lambda(y) * y, where lambda(y) = (-1 + sqrt(4<y, Ay> + 1)) / (2<y, Ay>)
        # we are interested in the inverse of the function h(x) = 2Ax / (1 - <x, Ax>)
        # note that h(x) = j(g(x)), where j(z) = 2Az.
        # therefore, inv(h)(y) = inv(g)(inv(j)(y))
        # NOTE: if <y, inv(A)y> is too small, or 0, we'll run into numerical issues.
        # so, we're artificially clipping it.
        yinvAy = torch.clamp_min(self._inverse_ellipsoid_inner_product(y), min=1e-08)
        lambda_cons = (-1 + torch.sqrt(1 + yinvAy)) / yinvAy
        return lambda_cons.unsqueeze(dim=-1) * self._inverse_ellipsoid_map(y)

    def hessian(self, x: torch.Tensor):
        """
        Computes the square root of the Hessian of the potential at x
        defined as (2A / (1 - <x, Ax>) + 4 (Ax)(Ax).T / (1 - <x, Ax>)^2).
        This has a nice structure which we could exploit for the square root,
        but not doing it here.
        """
        # NOTE: when x is too close to the boundary
        # this might be really small and cause problems for us when
        # dividing by 1 - <x, Ax>.
        one_minus_xAx = 1 - self._ellipsoid_inner_product(x)  # batchsize
        one_minus_xAx.clamp_min_(min=1e-08)
        Ax = self._ellipsoid_map(x)
        U = self.ellipsoid["rot"]
        L = self.ellipsoid["eigvals"]
        scaled_Ax = 2 * Ax / one_minus_xAx.unsqueeze_(dim=-1)
        scaled_L = 2 * L / one_minus_xAx.unsqueeze_(dim=-1)
        ULUT = torch.einsum("...ij,jk->...ik", U * scaled_L, U.T)
        return ULUT + scaled_Ax.unsqueeze(dim=-1) * scaled_Ax.unsqueeze(dim=-2)

    def sample_uniform(self, n_points: int):
        """
        Sampling points uniformly from an ellipsoid.
        This is just rejection sampling, so is very inefficient in higher dimensions
        volume of cuboid with side lengths [2/a_{i}]_{i = 1}^{d} is 2^{d} / prod([a_{i}]_{i = 1}^{d})
        volume of ellipsoid with axis lengths [1/a_{i}] is pi / prod([a_{i}]^{i = 1}^{d})
        So there's a pi /2^{d} chance that a point sampled within a cuboid lies in the ellipsoid
        """
        dimension = self.ellipsoid["eigvals"].shape[0]
        fraction = 2**dimension / torch.pi
        all_points = torch.empty(0, dimension)
        while all_points.shape[0] < n_points:
            # a little excess just in case
            points = torch.rand(int(1.1 * fraction * n_points), dimension) * 2 - 1
            points = torch.einsum("ij,...i->...j", self.ellipsoid["rot"], points)
            points = points * torch.sqrt(self.ellipsoid["eigvals"])
            feasible = self.feasibility(points)
            all_points = torch.vstack([all_points, points[feasible]])
        return all_points[:n_points]

    def boundary_to_interior_half(self, x: torch.Tensor):
        """
        Returns if x belongs in the region between the
        boundary and the interior where the volume of this
        region is 0.5 under the uniform distribution
        """
        return self._ellipsoid_inner_product(x) > (0.5 ** (2 / self.dimension))


class SimplexBarrier(Barrier):
    """
    Log barrier for Simplex
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def _safe_interior(self, x: torch.Tensor, squeeze_last_dim: bool = True):
        return torch.clamp_min(
            1 - torch.sum(x, dim=-1, keepdim=not squeeze_last_dim),
            min=1e-08,
        )

    def feasibility(self, x: torch.Tensor):
        sum_one_cons = torch.sum(x, dim=-1) <= 1
        pos_cons = torch.all(x >= 0, dim=-1)
        return sum_one_cons.logical_and_(pos_cons)

    def value(self, x: torch.Tensor):
        return -torch.sum(torch.log(x), dim=-1) - torch.log(
            self._safe_interior(x, True)
        )

    def gradient(self, x: torch.Tensor):
        x_safe = torch.clamp_min(x, min=1e-08)
        return -1 / x_safe + torch.reciprocal(self._safe_interior(x, False))

    def hessian(self, x: torch.Tensor):
        one_over_x_squared = torch.square(x).reciprocal_().clamp_max_(max=1e08)
        cons = (
            self._safe_interior(x, False).reciprocal_().square_().clamp_max_(max=1e08)
        )
        hess = torch.diag_embed(one_over_x_squared) + cons.unsqueeze_(-1)
        return hess

    def inverse_gradient(self, y: torch.Tensor):
        # assume shape of y is  (n, d)
        # here, we have to solve this numerically.
        # this is 1-D problem, which we will solve using binary search
        c_lower = torch.zeros(*y.shape[:-1])
        c_upper = torch.reciprocal(torch.max(y, dim=-1)[0])
        c_upper.clamp_max_(1)
        c_upper[c_upper < 0] = 1

        def f(c):
            return c + c * torch.sum(1 / (1 - c.unsqueeze(-1) * y), dim=-1) - 1

        for _ in range(35):  # 35 iterations should be sufficient
            c_middle = 0.5 * (c_lower + c_upper)
            f_middle = f(c_middle)
            mask = f_middle > 0
            c_upper[mask] = c_middle[mask]
            c_lower[~mask] = c_middle[~mask]
        c_upper = c_upper.unsqueeze(-1)

        return torch.clamp_min(c_upper / (1 - c_upper * y), min=1e-08)

    def boundary_to_interior_half(self, x: torch.Tensor):
        a = 0.5 ** (1 / self.dimension)
        value_of_mask = a - torch.sum(x, dim=-1)
        return value_of_mask < 0


class PolytopeBarrier(Barrier):
    """
    Log barrier for Polytope
    """

    def __init__(self, polytope: Dict[str, torch.Tensor]):
        self.polytope = polytope
        self.dimension = polytope["A"].shape[-1]
        if len(polytope["A"].shape) == 1:  # diagonal A
            self.diag_hess = True

    def _Ax(self, x: torch.Tensor):
        # functionality to compute Ax for batch of x
        if not self.diag_hess:
            return torch.einsum("ij,...j->...i", self.polytope["A"], x)
        else:
            return self.polytope["A"] * x

    def _safe_slack(self, x: torch.Tensor):
        vals = self._Ax(x=x)
        return torch.clamp_min(self.polytope["b"] - vals, min=1e-08)

    def feasibility(self, x: torch.Tensor):
        return torch.all(
            self._Ax(x=x) <= self.polytope["b"],
            dim=-1,
        )

    def value(self, x: torch.Tensor):
        return -torch.sum(torch.log(self._safe_slack(x)), dim=-1)

    def gradient(self, x: torch.Tensor):
        slack = self._safe_slack(x)  # [b - aTx]
        if not self.diag_hess:
            return torch.sum(self.polytope["A"] / slack.unsqueeze(-1), dim=-2)
        else:
            return self.polytope["A"] / slack

    def hessian(self, x: torch.Tensor):
        slack = self._safe_slack(x)  # [b - aTx]
        if not self.diag_hess:
            component = self.polytope["A"] / slack.unsqueeze(-1)  # shape n x m x d
            # component.unsqueeze(-1) * component.unsqueeze(-2) is of shape n x m x d x d
            return torch.sum(component.unsqueeze(-1) * component.unsqueeze(-2), dim=-3)
        else:
            return torch.square(self.polytope["A"] / slack)

    def inverse_gradient(self, x: torch.Tensor):
        # TODO: this seems hard
        raise NotImplementedError
