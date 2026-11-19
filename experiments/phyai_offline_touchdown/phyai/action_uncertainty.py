"""Action-conditional local projection of the full 87D uncertainty.

Finite-difference sensitivities are numerical evidence. The analytic remainder
function requires externally justified region, Hessian, model and numeric bounds.
"""
from dataclasses import dataclass
import math
import torch
from frozen.core.phi_rocket_physics import state_plus
from frozen.core.phi_rocket_types import STATE_SCALE
from .touchdown_decoder import rollout, prefix_margins


@dataclass
class Projection:
    factor: torch.Tensor
    whitened: torch.Tensor
    projector: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    tail_norm: torch.Tensor


def local_projection(jacobian, covariance, rank):
    if covariance.shape != (87, 87) or jacobian.shape[-1] != 87:
        raise ValueError("projection must retain the full 87D joint layout")
    if not torch.isfinite(jacobian).all() or not torch.isfinite(covariance).all():
        raise ValueError("nonfinite projection inputs")
    if not 0 <= rank <= min(jacobian.shape):
        raise ValueError("rank outside sensitivity dimensions")
    if not torch.allclose(covariance, covariance.T, atol=1e-8, rtol=1e-6):
        raise ValueError("covariance must be symmetric")
    eigen, vectors = torch.linalg.eigh(covariance.detach())
    if eigen.min() < -1e-8:
        raise ValueError("indefinite covariance")
    # Symmetric PSD square root preserves cross terms and coordinate identity.
    L = (vectors * eigen.clamp_min(0).sqrt()[None]) @ vectors.T
    A = jacobian @ L
    _, singular, vh = torch.linalg.svd(A, full_matrices=False)
    basis = vh[:rank].T
    P = basis @ basis.T
    tail = torch.linalg.vector_norm(A - A @ P, dim=-1)
    return Projection(L, A, P, basis, singular, tail)


def remainder_bound(projection, radius, hessian_bound, model_bound, numeric_bound):
    """Transfer an exact NONLINEAR projected lower bound to the full ball.

    With ||H|| <= h throughout the whitened ball, both Taylor remainders
    R(eta) and R(P eta) contribute: |R(eta)-R(P eta)| <= h*rho**2.
    The smaller .5*h*rho**2 applies to an affine surrogate lower bound,
    not to the nonlinear projected minimum accepted by this interface.
    Model/numeric terms must bound the full comparison error independently.
    """
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("invalid region radius")
    if any(x is None for x in (hessian_bound, model_bound, numeric_bound)):
        return torch.full_like(projection.tail_norm, float("inf"))
    bounds = [torch.as_tensor(x, device=projection.tail_norm.device, dtype=projection.tail_norm.dtype)
              for x in (hessian_bound, model_bound, numeric_bound)]
    if any(x.shape != projection.tail_norm.shape or not torch.isfinite(x).all() or (x < 0).any() for x in bounds):
        raise ValueError("bounds must be nonnegative finite vectors for all margins")
    h, model, numeric = bounds
    return radius * projection.tail_norm + h * radius**2 + model + numeric


def bound_status(projected_lower, error, *, region_qualified, modes_covered, projected_minimum_qualified):
    if not (region_qualified and modes_covered and projected_minimum_qualified):
        return "insufficient_evidence"
    if projected_lower.shape != error.shape or (error < 0).any():
        raise ValueError('error must be a nonnegative bound for every margin')
    if not torch.isfinite(projected_lower).all() or not torch.isfinite(error).all():
        return "insufficient_evidence"
    return "conditional_pass" if bool((projected_lower >= error).all()) else "insufficient_evidence"


def physical_sensitivity(physics, snapshot, controls, arrival, step=.002):
    """Same fixed command prefix in every perturbed world; no true-world decoder."""
    with torch.no_grad():
        x, theta = snapshot.state, snapshot.theta
        e = torch.eye(87, dtype=x.dtype, device=x.device) * step
        e = torch.cat((e, -e), 0)
        states = state_plus(x.expand(174, -1), e[:, :20] * x.new_tensor(STATE_SCALE))
        parameters = theta[None] + e[:, 20:]
        seq = controls.detach()[None].expand(174, -1, -1)
        out = rollout(physics, states, seq, parameters, arrival.detach().expand(174))
        margins = prefix_margins(out)
        J = ((margins[:87] - margins[87:]) / (2 * step)).T
    if not torch.isfinite(J).all():
        raise ValueError("nonfinite physical sensitivity")
    return J
