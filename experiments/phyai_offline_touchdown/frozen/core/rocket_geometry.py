"""Unchanged SO(3) residual extracted from the earlier model module."""
import torch

def rotation_error(predicted, actual):
    """SO(3) logarithm of actual^-1 predicted; invariant to quaternion sign."""
    av, aw = -actual[..., :3], actual[..., 3:4]
    bv, bw = predicted[..., :3], predicted[..., 3:4]
    xyz = aw*bv+bw*av+torch.cross(av,bv,dim=-1)
    w = aw*bw-(av*bv).sum(-1,keepdim=True)
    sign = torch.where(w<0,-1.,1.);xyz,w=xyz*sign,w*sign
    norm=xyz.norm(dim=-1,keepdim=True)
    scale=2*torch.atan2(norm,w.clamp_min(1e-8))/norm.clamp_min(1e-8)
    return xyz*torch.where(norm>1e-6,scale,2*torch.ones_like(scale))
