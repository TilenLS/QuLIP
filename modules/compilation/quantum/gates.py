import torch

def get_rotation_matrices(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    phase = phase / 2.0  # standard quantum mechanical rotation convention
    cos = torch.cos(phase).to(torch.complex64)
    sin = torch.sin(phase).to(torch.complex64)
    return cos, sin

def Rx(phase: torch.Tensor) -> torch.Tensor:
    cos, sin = get_rotation_matrices(phase)
    row0 = torch.stack([cos, -1j * sin], dim=-1)
    row1 = torch.stack([-1j * sin, cos], dim=-1)
    return torch.stack([row0, row1], dim=-2)

def Ry(phase: torch.Tensor) -> torch.Tensor:
    cos, sin = get_rotation_matrices(phase)
    row0 = torch.stack([cos, sin], dim=-1)
    row1 = torch.stack([-sin, cos], dim=-1)
    return torch.stack([row0, row1], dim=-2)

def Rz(phase: torch.Tensor) -> torch.Tensor:
    angle = phase / 2.0
    exp_neg = torch.exp(-1j * angle).to(torch.complex64)
    exp_pos = torch.exp(1j * angle).to(torch.complex64)
    zeros = torch.zeros_like(angle)
    row0 = torch.stack([exp_neg, zeros], dim=-1)
    row1 = torch.stack([zeros, exp_pos], dim=-1)
    return torch.stack([row0, row1], dim=-2)

def _build_controlled_gate(gate_fn, phase: torch.Tensor) -> torch.Tensor:
    target_mat = gate_fn(phase)
    batch_shape = list(phase.shape)
    device = phase.device

    ones = torch.ones(batch_shape + [1], dtype=torch.complex64, device=device)
    zeros = torch.zeros(batch_shape + [1], dtype=torch.complex64, device=device)
    zeros_2 = torch.zeros(batch_shape + [2], dtype=torch.complex64, device=device)

    row0 = torch.cat([ones, zeros, zeros_2], dim=-1)
    row1 = torch.cat([zeros, ones, zeros_2], dim=-1)
    row2 = torch.cat([zeros_2, target_mat[..., 0, :]], dim=-1)
    row3 = torch.cat([zeros_2, target_mat[..., 1, :]], dim=-1)
    full_mat = torch.stack([row0, row1, row2, row3], dim=-2)
    
    return full_mat.view(batch_shape + [2, 2, 2, 2])

def CRx(phase: torch.Tensor) -> torch.Tensor: return _build_controlled_gate(Rx, phase)
def CRy(phase: torch.Tensor) -> torch.Tensor: return _build_controlled_gate(Ry, phase)
def CRz(phase: torch.Tensor) -> torch.Tensor: return _build_controlled_gate(Rz, phase)