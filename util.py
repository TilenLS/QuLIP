import torch
    
def fs_distance(state1, state2):
    inner_product = torch.sum(state1 * state2.conj(), dim=1)
    return (torch.pi - torch.acos(inner_product.abs().clamp(0,1)))

def qcosine(bstates1, bstates2, eps=1e-9):
    norm1 = torch.linalg.vector_norm(bstates1, ord=2, dim=1)
    norm2 = torch.linalg.vector_norm(bstates2, ord=2, dim=1)
    inner_product = torch.sum(bstates1 * bstates2.conj(), dim=1)
    return inner_product.abs() / (norm1 * norm2 + eps)

def Rz(phase):
    half_theta = torch.pi * phase
    exp1 = torch.exp(-1j * half_theta)
    exp2 = torch.exp(1j * half_theta)

    row1 = torch.stack([exp1, torch.zeros_like(exp1)])
    row2 = torch.stack([torch.zeros_like(exp2), exp2])
    return torch.stack([row1, row2])

def BatchRz(phase):
    half_thetas = torch.pi * phase

    exp_neg = torch.exp(-1j * half_thetas)
    exp_pos = torch.exp(1j * half_thetas)

    zeros = torch.zeros_like(exp_neg)

    return torch.stack([torch.stack([exp_neg, zeros], dim=-1), torch.stack([zeros, exp_pos], dim=-1)], dim=-2)


def Rx(phase):
    half_theta = torch.pi * phase
    sin = -1j*torch.sin(half_theta)
    cos = torch.cos(half_theta)

    row1 = torch.stack([cos, sin])
    row2 = torch.stack([sin, cos])
    return torch.stack([row1, row2])

def BatchRx(phase):
    half_theta = torch.pi * phase
    cos = torch.cos(half_theta)
    sin = -1j * torch.sin(half_theta)
    return torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)

def CRz(phase):
    return torch.block_diag(torch.eye(2), Rz(phase)).reshape(2,2,2,2)

def BatchCRz(phase):
    B = phase.shape[0]
    rz_part = BatchRz(phase)
    eye_part = torch.eye(2).unsqueeze(0).expand(B, -1, -1)
    zero_part = torch.zeros(B, 2, 2)
    top_row = torch.cat([eye_part, zero_part], dim=2)
    bottom_row = torch.cat([zero_part, rz_part], dim=2)
    full_mat = torch.cat([top_row, bottom_row], dim=1)
    return full_mat.view(B, 2, 2, 2, 2)