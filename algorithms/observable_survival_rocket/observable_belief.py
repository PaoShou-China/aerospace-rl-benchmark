"""Causal, full-covariance physical parameter inference.

An extended Kalman *approximation*, not a calibrated posterior or a safety
certificate. No task reward enters this update. Jacobians and innovations must
be computed from a real executed-action transition, before assimilation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class BeliefUpdate:
    mean: Tensor
    covariance: Tensor
    innovation_nis: Tensor
    observable_rank: Tensor
    accepted: Tensor


def joseph_update(mean: Tensor, covariance: Tensor, innovation: Tensor,
                  jacobian: Tensor, observation_variance: Tensor,
                  valid: Tensor, *, process_variance: float = 1e-6,
                  innovation_clip: float = 5.0) -> BeliefUpdate:
    """One batched update; invalid observations leave mean AND covariance intact.

    Shapes: mean [B,P], covariance [B,P,P], innovation [B,D], Jacobian [B,D,P],
    observation_variance [B,D], valid [B]. Rank is diagnostic only. A missing
    sensitivity does not create evidence. Joseph form preserves PSD up to roundoff.
    """
    if mean.ndim != 2 or innovation.ndim != 2:
        raise ValueError('mean and innovation must be rank-two batches')
    batch, width = mean.shape
    dim = innovation.shape[1]
    expected = ((batch, width, width), (batch, dim, width), (batch, dim), (batch,))
    actual = (covariance.shape, jacobian.shape, observation_variance.shape, valid.shape)
    if any(tuple(a) != e for a, e in zip(actual, expected)):
        raise ValueError(f'incompatible belief shapes: {actual}, expected {expected}')
    if process_variance < 0 or innovation_clip <= 0:
        raise ValueError('process variance must be nonnegative and clip positive')
    if not bool(torch.isfinite(mean).all() and torch.isfinite(covariance).all()):
        raise ValueError('nonfinite prior belief')
    if not bool((observation_variance > 0).all()):
        raise ValueError('observation variance must be positive')
    finite = (torch.isfinite(innovation).all(-1)
              & torch.isfinite(jacobian).all((-1, -2))
              & torch.isfinite(observation_variance).all(-1))
    mask = valid.bool() & finite
    eye = torch.eye(width, device=mean.device, dtype=mean.dtype).expand(batch, -1, -1)
    prior = (covariance + covariance.transpose(-1, -2)) * .5 + process_variance * eye
    # Sanitization occurs before batched factorizations; invalid data are never used.
    h = torch.where(mask[:, None, None], jacobian, torch.zeros_like(jacobian))
    innovation = torch.where(mask[:, None], innovation, torch.zeros_like(innovation))
    variance = torch.where(mask[:, None], observation_variance,
                           torch.ones_like(observation_variance))
    noise = torch.diag_embed(variance)
    ph = prior @ h.transpose(-1, -2)
    s = h @ ph + noise
    chol, info = torch.linalg.cholesky_ex(s)
    if bool((info != 0).any()):
        raise ValueError('innovation covariance is not positive definite')
    gain = torch.cholesky_solve(ph.transpose(-1, -2), chol).transpose(-1, -2)
    whitened = torch.cholesky_solve(innovation[..., None], chol).squeeze(-1)
    nis = (innovation * whitened).sum(-1)
    # Robustify only the mean step. This clipping makes Gaussian coverage an
    # empirical quantity: do not market nominal intervals as guarantees.
    standard = s.diagonal(dim1=-2, dim2=-1).sqrt()
    clipped = innovation.clamp(-innovation_clip * standard, innovation_clip * standard)
    updated_mean = mean + (gain @ clipped[..., None]).squeeze(-1)
    complement = eye - gain @ h
    updated_covariance = complement @ prior @ complement.transpose(-1, -2)
    updated_covariance = updated_covariance + gain @ noise @ gain.transpose(-1, -2)
    updated_covariance = (updated_covariance + updated_covariance.transpose(-1, -2)) * .5
    # No arbitrary eigenvalue floor is added after assimilation: that would
    # produce artificial evidence or covariance contraction in null directions.
    rank = torch.linalg.matrix_rank(h / variance.sqrt()[..., None], rtol=1e-4)
    return BeliefUpdate(
        torch.where(mask[:, None], updated_mean, mean),
        torch.where(mask[:, None, None], updated_covariance, covariance),
        torch.where(mask, nis, torch.zeros_like(nis)), rank, mask)


def cubature_points(mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
    """2P+1 positive-weight points matching the FULL prior covariance.

    Center weight 1/3, other weights 1/(3P). Unlike three collinear hypotheses,
    this rule represents all parameter directions. These are integration points,
    not posterior samples and not confidence bounds.
    """
    p = mean.shape[-1]
    symmetric = (covariance + covariance.transpose(-1, -2)) * .5
    chol, info = torch.linalg.cholesky_ex(symmetric)
    if bool((info != 0).any()):
        raise ValueError('cubature requires positive definite covariance')
    offsets = (1.5 * p) ** .5 * chol.transpose(-1, -2)
    points = torch.cat((mean[:, None], mean[:, None] + offsets,
                        mean[:, None] - offsets), dim=1)
    weights = mean.new_full((mean.shape[0], 2 * p + 1), 1 / (3 * p))
    weights[:, 0] = 1 / 3
    return points, weights


def batch_jacobian(outputs: Tensor, inputs: Tensor) -> Tensor:
    """For independent batch members only; never use on batch-coupled models."""
    rows = [torch.autograd.grad(outputs[:, j].sum(), inputs, retain_graph=True,
                                create_graph=False)[0]
            for j in range(outputs.shape[-1])]
    return torch.stack(rows, dim=1)
