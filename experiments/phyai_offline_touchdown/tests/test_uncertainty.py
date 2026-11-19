import torch
import math
from phyai.action_uncertainty import local_projection, remainder_bound, bound_status


def test_projection_is_action_dependent_and_keeps_cross_covariance():
    cov = torch.eye(87, dtype=torch.float64)
    cov[0, 1] = cov[1, 0] = .6
    a = torch.zeros(2, 87, dtype=torch.float64); a[0, 0] = 2; a[1, 7] = .1
    b = torch.zeros_like(a); b[0, 12] = 2; b[1, 7] = .1
    pa, pb = local_projection(a, cov, rank=1), local_projection(b, cov, rank=1)
    assert pa.projector.shape == (87, 87)
    assert not torch.allclose(pa.projector, pb.projector)
    assert torch.allclose(pa.whitened @ pa.whitened.T, a @ cov @ a.T)
    assert torch.allclose(pa.projector @ pa.projector, pa.projector, atol=1e-10)
    assert pa.tail_norm[1] > .09


def test_omitted_linear_direction_cannot_silently_pass():
    a = torch.zeros(2, 87, dtype=torch.float64); a[0, 0] = 3; a[1, 1] = 2
    p = local_projection(a, torch.eye(87, dtype=torch.float64), rank=1)
    error = remainder_bound(p, radius=.5, hessian_bound=torch.zeros(2), model_bound=torch.zeros(2), numeric_bound=torch.zeros(2))
    assert torch.allclose(error, torch.tensor([0., 1.],dtype=torch.float64))
    margin = torch.tensor([1., .5],dtype=torch.float64)
    assert bound_status(margin, error, region_qualified=True, modes_covered=True, projected_minimum_qualified=True) == "insufficient_evidence"


def test_missing_or_unqualified_bounds_never_issue_conditional_pass():
    p = local_projection(torch.eye(87,dtype=torch.float64)[:2], torch.eye(87,dtype=torch.float64), 2)
    e = remainder_bound(p, 1, None, None, None)
    assert torch.isinf(e).all()
    assert bound_status(torch.ones(2), torch.zeros(2), region_qualified=False, modes_covered=True,
                        projected_minimum_qualified=True) == "insufficient_evidence"
    assert bound_status(torch.ones(2), torch.zeros(2), region_qualified=True, modes_covered=False,
                        projected_minimum_qualified=True) == "insufficient_evidence"
    assert bound_status(torch.ones(2), torch.zeros(2), region_qualified=True, modes_covered=True,
                        projected_minimum_qualified=False) == "insufficient_evidence"


def test_nonlinear_projected_minimum_needs_both_taylor_remainders():
    # Exact 2D quadratic embedded in 87D. H has eigenvalues +/-1. The
    # projected minimum is positive but an omitted-direction point is unsafe.
    J = torch.zeros(1,87,dtype=torch.float64); J[0,0] = -1/3
    proj = local_projection(J,torch.eye(87,dtype=torch.float64),1)
    H = torch.tensor([[1/math.sqrt(3),-math.sqrt(2/3)],
                      [-math.sqrt(2/3),-1/math.sqrt(3)]],dtype=torch.float64)
    point = torch.tensor([1/math.sqrt(3),math.sqrt(2/3)],dtype=torch.float64)
    assert torch.allclose(torch.linalg.matrix_norm(H,ord=2),torch.tensor(1.,dtype=torch.float64))
    assert .65-point[0]/3+.5*point@H@point < 0
    projected_min = torch.tensor([.65-1/(6*math.sqrt(3))],dtype=torch.float64)
    error = remainder_bound(proj,1.,torch.ones(1),torch.zeros(1),torch.zeros(1))
    assert bound_status(projected_min,error,region_qualified=True,modes_covered=True,
                        projected_minimum_qualified=True) == 'insufficient_evidence'
