from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from collections import defaultdict
import numpy as np

def tn2qiskit(einsum_expr, gate_arr):
    input_indices, out_list = einsum_expr
    nq = sum(1 for _, gate in gate_arr if gate == '0')
    qc = QuantumCircuit(nq, nq)
    wire2q = defaultdict(list)
    qcounter = 0
    param_dict = {}

    for idx_arr, (symbol, gate) in zip(input_indices, gate_arr):
        if gate == '0':
            wire_name = idx_arr[0]
            wire2q[wire_name].append(qcounter)
            qcounter += 1
        else: 
            num_in = len(idx_arr) // 2 
            in_wires = idx_arr[:num_in]
            out_wires = idx_arr[num_in:]
            q_targets = [wire2q[w].pop(0) for w in in_wires]

            gate_func = getattr(qc, gate.lower())
            if symbol is None:
                gate_func(*q_targets)
            else:
                new_param = Parameter(symbol)
                gate_func(new_param, *q_targets)
                param_dict[new_param] = np.random.rand() * 2 * np.pi
            
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