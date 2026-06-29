import torch, math
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from collections import defaultdict
import numpy as np

def fs_distance(state1, state2):
    inner_product = torch.sum(state1 * state2.conj(), dim=1)
    return (torch.full(inner_product.size(), torch.pi/2) - torch.acos(inner_product.abs().clamp(0,1)))

def qcosine(bstates1, bstates2, eps=1e-9):
    norm1 = torch.linalg.vector_norm(bstates1, ord=2, dim=1)
    norm2 = torch.linalg.vector_norm(bstates2, ord=2, dim=1)
    inner_product = torch.sum(bstates1 * bstates2.conj(), dim=1)
    return inner_product.abs() / (norm1 * norm2 + eps)

def amplitude_encoding(vector):
    dim = len(vector)
    num_qubits = math.ceil(math.log2(dim))
    target_dim = 2 ** num_qubits


    if dim < target_dim:
        padding_size = target_dim - dim
        vector = np.concatenate([vector, np.zeros(padding_size)])

    norm = np.linalg.norm(vector, ord=2)
    if norm > 0:
        state_vector = vector / norm
    else:
        state_vector = torch.zeros_like(vector)
        state_vector[0] = 1.0
    
    return state_vector

def tn2qiskit(einsum_expr, gate_arr):
    input_indices, _ = einsum_expr
    nq = sum(1 for gate in gate_arr if gate['op_type'] == '0')
    qc = QuantumCircuit(nq, nq)
    wire2q = defaultdict(list)
    qcounter = 0
    param_dict = {}
    name2param = {}

    for idx_arr, gate in zip(input_indices, gate_arr):
        if gate['op_type'] == '0':
            wire_name = idx_arr[0]
            wire2q[wire_name].append(qcounter)
            qcounter += 1
        elif gate['op_type'] == '0_dag':
            wire2q[idx_arr[0]].pop()
        else: 
            num_in = len(idx_arr) // 2 
            in_wires = idx_arr[:num_in]
            out_wires = idx_arr[num_in:]
            q_targets = [wire2q[w].pop(0) for w in in_wires]

            gate_func = getattr(qc, gate['op_type'].lower())
            if gate['name'] is None:
                gate_func(*q_targets)
            else:
                if gate['name'] in name2param:
                    param = name2param[gate['name']]
                else:
                    param = Parameter(gate['name'])
                    name2param[gate['name']] = param
                gate_func(param, *q_targets)
                param_dict[param] = np.random.rand() * 2 * np.pi

            for w, q in zip(out_wires, q_targets):
                wire2q[w].append(q) 

    output_qubits = []
    for wire_name, q_list in wire2q.items():
        if len(q_list) == 2:
            q_left, q_right = q_list
            qc.cx(q_left, q_right)
            qc.h(q_left)
            qc.measure(q_left, q_left)
            qc.measure(q_right, q_right)
        elif len(q_list) == 1:
            q_out = q_list[0]
            qc.measure(q_out, q_out)
            output_qubits.append(q_out)
    
    return qc, output_qubits, param_dict