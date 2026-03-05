import torch, random
import numpy as np

def max_width(einsum_str, tarr):
    from opt_einsum import contract_path
    shapes = []
    for i, (symbol, op_type) in enumerate(tarr):
        if symbol is None:
            if op_type == 'sqrt':
                data = torch.tensor(2.0**0.5, dtype=torch.complex64)
            elif op_type == '0':
                data = torch.tensor([1,0], dtype=torch.complex64)
            elif op_type == 'H':
                data = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64) / (2.0**0.5)
            elif op_type == 'CX':
                data = torch.block_diag(torch.eye(2), torch.tensor([[0.0,1.0],[1.0,0.0]], dtype=torch.complex64)).reshape(2,2,2,2)
            shapes.append(data.shape)
        else:
            if op_type == 'Rz' or op_type == 'Rx' or op_type == 'Ry':
                shapes.append(torch.Size([2, 2]))
            if op_type == 'CRz' or op_type == 'CRx' or op_type == 'CRy': 
                shapes.append(torch.Size([2, 2, 2, 2]))
    path_info = contract_path(einsum_str, *shapes, shapes=True)
    return path_info[1].largest_intermediate
    
def fs_distance(state1, state2):
    inner_product = torch.sum(state1 * state2.conj(), dim=1)
    return (torch.full(inner_product.size(), torch.pi/2) - torch.acos(inner_product.abs().clamp(0,1)))

def qcosine(bstates1, bstates2, eps=1e-9):
    norm1 = torch.linalg.vector_norm(bstates1, ord=2, dim=1)
    norm2 = torch.linalg.vector_norm(bstates2, ord=2, dim=1)
    inner_product = torch.sum(bstates1 * bstates2.conj(), dim=1)
    return inner_product.abs() / (norm1 * norm2 + eps)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def Rz(phase, dtype=torch.complex64):
    #half_theta = torch.pi * phase

    exp1 = torch.exp(-1j * phase).to(dtype)
    exp2 = torch.exp(1j * phase).to(dtype)

    return torch.tensor([[exp1, 0], [0, exp2]], dtype=dtype)

    # row1 = torch.stack([exp1, torch.zeros_like(exp1)])
    # row2 = torch.stack([torch.zeros_like(exp2), exp2])
    # return torch.stack([row1, row2])

def BatchRz(phase, dtype=torch.complex64):
    # half_thetas = torch.pi * phase

    exp_neg = torch.exp(-1j * phase).to(dtype)
    exp_pos = torch.exp(1j * phase).to(dtype)

    B = phase.shape[0]
    gate = torch.zeros((B, 2, 2), dtype=dtype, device=phase.device)
    gate[:, 0, 0] = exp_neg
    gate[:, 1, 1] = exp_pos

    return gate

    # zeros = torch.zeros_like(exp_neg)
    # return torch.stack([torch.stack([exp_neg, zeros], dim=-1), torch.stack([zeros, exp_pos], dim=-1)], dim=-2)

def Rx(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    sin = -1j*torch.sin(phase).to(dtype)
    cos = torch.cos(phase).to(dtype)

    return torch.tensor([[cos, sin], [sin, cos]], dtype=dtype)

    # row1 = torch.stack([cos, sin])
    # row2 = torch.stack([sin, cos])
    # return torch.stack([row1, row2])

def BatchRx(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    cos = torch.cos(phase).to(dtype)
    sin = -1j * torch.sin(phase).to(dtype)

    B = phase.shape[0]
    gate = torch.empty((B, 2, 2), dtype=dtype, device=phase.device)
    gate[:, 0, 0] = cos
    gate[:, 0, 1] = sin
    gate[:, 1, 0] = sin
    gate[:, 1, 1] = cos

    return gate

    # return torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)

def Ry(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    sin = torch.sin(phase).to(dtype)
    cos = torch.cos(phase).to(dtype)

    return torch.tensor([[cos, sin], [-sin, cos]], dtype=dtype)

    # row1 = torch.stack([cos, sin])
    # row2 = torch.stack([-sin, cos])
    # return torch.stack([row1, row2])

def BatchRy(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    cos = torch.cos(phase).to(dtype)
    sin = torch.sin(phase).to(dtype)

    B = phase.shape[0]
    gate = torch.empty((B, 2, 2), dtype=dtype, device=phase.device)
    gate[:, 0, 0] = cos
    gate[:, 0, 1] = sin
    gate[:, 1, 0] = -sin
    gate[:, 1, 1] = cos

    return gate

    # return torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=-2)

def CRz(phase):
    return torch.block_diag(torch.eye(2), Rz(phase)).reshape(2,2,2,2)

def BatchCRz(phase, dtype=torch.complex64):
    B = phase.shape[0]

    full_mat = torch.zeros((B, 4, 4), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = 1.0
    full_mat[:, 1, 1] = 1.0
    full_mat[:, 2:, 2:] = BatchRz(phase)

    # rz_part = BatchRz(phase)
    # eye_part = torch.eye(2).unsqueeze(0).expand(B, -1, -1)
    # zero_part = torch.zeros(B, 2, 2)
    # top_row = torch.cat([eye_part, zero_part], dim=2)
    # bottom_row = torch.cat([zero_part, rz_part], dim=2)
    # full_mat = torch.cat([top_row, bottom_row], dim=1)

    return full_mat.view(B, 2, 2, 2, 2)

def CRx(phase):
    return torch.block_diag(torch.eye(2), Rx(phase)).reshape(2,2,2,2)

def BatchCRx(phase, dtype=torch.complex64):
    B = phase.shape[0]

    full_mat = torch.zeros((B, 4, 4), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = 1.0
    full_mat[:, 1, 1] = 1.0
    full_mat[:, 2:, 2:] = BatchRx(phase, dtype=dtype)

    # rx_part = BatchRx(phase)
    # eye_part = torch.eye(2).unsqueeze(0).expand(B, -1, -1)
    # zero_part = torch.zeros(B, 2, 2)
    # top_row = torch.cat([eye_part, zero_part], dim=2)
    # bottom_row = torch.cat([zero_part, rx_part], dim=2)
    # full_mat = torch.cat([top_row, bottom_row], dim=1)

    return full_mat.view(B, 2, 2, 2, 2)

def CRy(phase):
    return torch.block_diag(torch.eye(2), Ry(phase)).reshape(2,2,2,2)

def BatchCRy(phase, dtype=torch.complex64):
    B = phase.shape[0]

    full_mat = torch.zeros((B, 4, 4), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = 1.0
    full_mat[:, 1, 1] = 1.0
    full_mat[:, 2:, 2:] = BatchRy(phase, dtype=dtype)

    # ry_part = BatchRy(phase)
    # eye_part = torch.eye(2).unsqueeze(0).expand(B, -1, -1)
    # zero_part = torch.zeros(B, 2, 2)
    # top_row = torch.cat([eye_part, zero_part], dim=2)
    # bottom_row = torch.cat([zero_part, ry_part], dim=2)
    # full_mat = torch.cat([top_row, bottom_row], dim=1)

    return full_mat.view(B, 2, 2, 2, 2)

def map_phase(theta, rot_type=0, dtype=torch.complex64):
    from lambeq.backend.quantum import Controlled, Rz, Rx
    if rot_type == 1:
        return torch.tensor(Rz(theta).array, dtype=dtype)
    elif rot_type == 2:
        return torch.tensor(Rx(theta).array, dtype=dtype)
    elif rot_type == 3:
        return torch.tensor(Controlled(Rz(theta)).array, dtype=dtype)