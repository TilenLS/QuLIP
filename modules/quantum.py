from itertools import count
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

OUT_DIM = 9
dev = qml.device('default.qubit', wires=OUT_DIM)

class BaseAnsatz(ABC):
    def __init__(self, obmap=set(), layers=1):
        super().__init__()
        self.obmap = obmap
        self.layers = layers
        self.char_idx = count(0)

    def __call__(self, tn, curry=False):
        if curry:
            return self.tn2ansatz_curried(tn)
        else:
            return self.tn2ansatz(tn)

    def tns2ansatze(self, tn_arr, curry=False):
        return [self(tn, curry=curry) for tn in tn_arr]

    def reset_char(self):
        self.char_idx = count(0)

    def get_char(self):
        i = next(self.char_idx)
        return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] if i < 52 else chr(192 + i - 52)

    def ccg_map(self, tn):
        ccg_map = {}
        for _, idx_arr, type_arr in tn:
            for idx, typ in zip(idx_arr, type_arr):
                if idx not in ccg_map:
                    # Default to 1 qubit if type is missing from type_qubits
                    ccg_map[idx] = [self.get_char() for _ in range(self.obmap.get(typ, 1))]
        return ccg_map


    # def _gen_einsum_expr(self, input_indices, ccg_map):
    #     idx_counts = Counter("".join(input_indices))
    #     output_indices = [c for c, cnt in idx_counts.items() if cnt == 1]
        
    #     # Prioritize the root index (0) wires in the output string
    #     root_indices = ccg_map.get(0, [])
    #     out_str = "".join(c for c in root_indices if c in output_indices) + "".join(c for c in output_indices if c not in root_indices)
    #     return ",".join(input_indices) + "->" + out_str

    @abstractmethod
    def ansatz(self, current_wires, base_symbol):
        pass

    def gen_einsum_expr(self, input_indices, ccg_map):
        all_idx = [i for sub in input_indices for i in sub]
        idx_counts = Counter(all_idx)
        output_indices = [idx for idx, cnt in idx_counts.items() if cnt == 1]

        root_indices = ccg_map.get(0, ccg_map.get('0', []))
        out_list = [i for i in root_indices if i in output_indices] + \
                   [i for i in output_indices if i not in root_indices]
        
        return (input_indices, out_list)

    def _tn2ansatz(self, tn):
        self.reset_char()
        ccg_map = self.ccg_map(tn)
        input_indices, tensor_arr = [], []

        for word, idx_arr, type_arr in tn:
            out_wires = sum([ccg_map[idx] for idx in idx_arr], [])
            N = len(out_wires)
            current_wires = [self.get_char() for _ in range(N)]

            for w in current_wires:
                input_indices.append([w])
                tensor_arr.append((None, '0'))

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            replace_map = dict(zip(current_wires, out_wires))
            # input_indices = ["".join(replace_map.get(c, c) for c in inp) for inp in input_indices] 
            input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr
    
    def tn2ansatz(self, tn):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []

        for word, idx_arr, type_arr in tn:
            N = sum(self.obmap.get(typ, 1) for typ in type_arr)
            current_wires = [self.get_char() for _ in range(N)]

            for w in current_wires:
                input_indices.append([w])
                tensor_arr.append((None, '0'))

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            i = 0
            for idx, typ in zip(idx_arr, type_arr):
                n = self.obmap.get(typ, 1)
                if idx not in ccg_map:
                    ccg_map[idx] = current_wires[i:i+n]
                else:
                    target_wires = ccg_map[idx]                    
                    replace_map = dict(zip(current_wires[i:i+n], target_wires))
                    input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]
                i += n
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr

    def _tn2ansatz_curried(self, tn):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []
        tn_sorted = sorted(tn, key=lambda x: (0 in x[1], len(x[1])))

        for word, idx_arr, type_arr in tn_sorted:
            current_wires = []
            if len(idx_arr) == 1:
                n = self.obmap.get(type_arr[0], 1)
                current_wires.extend([self.get_char() for _ in range(n)])
                input_indices.extend([[w] for w in current_wires])
                tensor_arr.extend([(None, '0')] * n)
            else:
                elim_counter = defaultdict(int) 
                for idx, typ in zip(idx_arr, type_arr):
                    if idx in ccg_map.keys():
                        current_wires.extend(ccg_map[idx])
                        elim_counter[typ] += 1 
                    else:
                        elim_counter[typ] -= 1 
                if not any(elim_counter.values()):
                    pass 
                else:
                    for idx, typ in zip(idx_arr, type_arr):
                        if idx not in ccg_map.keys():
                            n = self.obmap.get(typ, 1)
                            new_wires = [self.get_char() for _ in range(n)]
                            current_wires.extend(new_wires)
                            input_indices.extend([[w] for w in new_wires])
                            tensor_arr.extend([(None, '0')] * n)

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend([[idx for idx in ten] for ten in new_indices])
            tensor_arr.extend(new_tensors)

            i = 0
            for idx, typ in zip(idx_arr, type_arr):
                n = self.obmap.get(typ, 1)
                ccg_map[idx] = current_wires[i:i+n]
                i += n

        for word, idx_arr, type_arr in tn_sorted:
            for idx, typ in zip(idx_arr, type_arr):
                if idx != 0:
                    n = self.obmap.get(typ, 1)
                    input_indices.extend([[w] for w in ccg_map[idx]])
                    tensor_arr.extend([(None, '0_dag')] * n)
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr


    def tn2ansatz_curried(self, tn):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []
        tn_sorted = sorted(tn, key=lambda x: (0 in x[1], len(x[1])))

        for word, idx_arr, type_arr in tn_sorted:
            current_wires = []
            for idx, typ in zip(idx_arr, type_arr):
                if idx in ccg_map:
                    current_wires.extend(ccg_map[idx])
                else:
                    n = self.obmap.get(typ, 1)
                    new_wires = [self.get_char() for _ in range(n)]
                    current_wires.extend(new_wires)
                    input_indices.extend([[w] for w in new_wires])
                    tensor_arr.extend([(None, '0')] * n)

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend([[idx for idx in ten] for ten in new_indices])
            tensor_arr.extend(new_tensors)

            i = 0
            for idx, typ in zip(idx_arr, type_arr):
                n = self.obmap.get(typ, 1)
                ccg_map[idx] = current_wires[i:i+n]
                i += n

        for virtual_idx, physical_wires in ccg_map.items():
            if virtual_idx != 0:
                input_indices.extend([[w] for w in physical_wires])
                tensor_arr.extend([(None, '0_dag')] * len(physical_wires))
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr

    # def tn2ansatz_curried(self, tn):
    #     self.reset_char() # I was here
    #     ccg_map = self.ccg_map(tn)
    #     input_indices, tensor_arr = [], []

    #     lifespans = Counter()
    #     for _, idx_arr, _ in tn:
    #         for idx in idx_arr:
    #             for gw in ccg_map[idx]:
    #                 lifespans[gw] += 1

    #     external_wires = {gw for gw, count in lifespans.items() if count == 1}
    #     free_qubits = []
    #     active_wires = {}
    #     cutwidth_counter = 0

    #     for word, idx_arr, type_arr in tn:
    #         global_wires = sum([ccg_map[idx] for idx in idx_arr], [])
    #         ansatz_inputs = []

    #         for gw in global_wires:
    #             if gw not in active_wires:
    #                 if free_qubits:
    #                     current_leg = free_qubits.pop()
    #                 else: 
    #                     current_leg = self.get_char()
    #                     cutwidth_counter += 1
    #                     input_indices.append([current_leg])
    #                     tensor_arr.append((None, '0'))

    #                 active_wires[gw] = current_leg
    #             ansatz_inputs.append(active_wires[gw])

    #         base_symbol = f"{word}__{'@'.join(type_arr)}"
    #         current_wires_out, new_indices, new_tensors = self.ansatz(ansatz_inputs, base_symbol)
    #         input_indices.extend(new_indices)
    #         tensor_arr.extend(new_tensors)

    #         for gw, out_leg in zip(global_wires, current_wires_out):
    #             active_wires[gw] = out_leg
    #             lifespans[gw] -= 1
    #             if lifespans[gw] == 0 and gw not in external_wires:
    #                 free_qubits.append(out_leg)
    #                 del active_wires[gw]

    #     replace_map = {leg: gw for gw, leg in active_wires.items()}
    #     input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]

    #     einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
    #     return einsum_expr, tensor_arr

class CustomV5Ansatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for i in range(N):
            nxt = self.get_char()
            new_indices.append(current_wires[i] + nxt)
            new_tensors.append((None, 'H'))
            current_wires[i] = nxt

        for l in range(self.layers):
            op_idx = 0
            for i in range(N):
                for g in ['Rz', 'Ry', 'Rz']:
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", g))
                    current_wires[i] = nxt
                    op_idx += 1
            if N > 1:
                for i in range(N):
                    c_idx, t_idx = i, (i + 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    new_tensors.append((None, 'CX'))
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
        return current_wires, new_indices, new_tensors
    
class IQPAnsatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for l in range(self.layers):
            op_idx = 0
            for i in range(N):
                nxt = self.get_char()
                new_indices.append(current_wires[i] + nxt)
                new_tensors.append((None, 'H'))
                current_wires[i] = nxt

                nxt = self.get_char()
                new_indices.append(current_wires[i] + nxt)
                new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Rz'))
                current_wires[i] = nxt
                op_idx += 1

            if N > 1:
                for i in range(N):
                    c_idx, t_idx = i, (i + 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRz'))
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1

        for i in range(N):
            nxt = self.get_char()
            new_indices.append(current_wires[i] + nxt)
            new_tensors.append((None, 'H'))
            current_wires[i] = nxt
            
        return current_wires, new_indices, new_tensors
        
class BrickworkAnsatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for l in range(self.layers):
            op_idx = 0
            if N > 1:
                for i in range(N):
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Ry'))
                    current_wires[i] = nxt
                    op_idx += 1

                for i in range(N-1):
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[i] + current_wires[i+1] + c_out + t_out)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRx'))
                    current_wires[i], current_wires[i+1] = c_out, t_out
                    op_idx += 1
            else:
                for g in ['Rz', 'Ry', 'Rz']:
                    nxt = self.get_char()
                    new_indices.append(current_wires[0] + nxt)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", g))
                    current_wires[0] = nxt
                    op_idx += 1

        return current_wires, new_indices, new_tensors

class Sim14Ansatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for l in range(self.layers):
            op_idx = 0
            if N > 1:
                for i in range(N):
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Ry'))
                    current_wires[i] = nxt
                    op_idx += 1
                for i in range(N):
                    c_idx, t_idx = i, (i - 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRx'))
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1

                for i in range(N):
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Ry'))
                    current_wires[i] = nxt
                    op_idx += 1
                for i in range(N, 0, -1):
                    c_idx, t_idx = i % N, (i-1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRx'))
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
            else:
                for g in ['Rx', 'Rz', 'Rx']:
                    nxt = self.get_char()
                    new_indices.append(current_wires[0] + nxt)
                    new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", g))
                    current_wires[0] = nxt
                    op_idx += 1

        return current_wires, new_indices, new_tensors
    

def tn2qc(tn_arr, ansatz, curry=False):
    qc_arr = []
    for tn_batch in tqdm(tn_arr):
        qc_batch = [ansatz(tn, curry=curry) for tn in tn_batch]
        qc_arr.append(qc_batch)
    return qc_arr
    
# def tn2qc_aro(tn_arr, ansatz, curry=False):
#     qc_arr = []
#     for pos_tn, neg_tn in tqdm(tn_arr):
#         pos_qc = ansatz(pos_tn, curry=curry)
#         neg_qc = ansatz(neg_tn, curry=curry)
#         qc_arr.append((pos_qc, neg_qc))
#     return qc_arr

# def tn2qc_svo(tn_arr, ansatz, curry=False):
#     qc_arr = []
#     for tn in tqdm(tn_arr):
#         qc = ansatz(tn, curry=curry)
#         qc_arr.append(qc)
#     return qc_arr


from sklearn.decomposition import PCA

class ImageFeatureMap(BaseAnsatz):
    def __init__(self, clip_dataset, k, obmap=set(), layers=2):
        super().__init__(obmap, layers)
        self.clip_dataset = clip_dataset
        self.k = k
        self.pca = PCA(n_components=k)
        self.pca.fit(clip_dataset)

    def compress_vector(self, raw_image_vector):
        flat_vector = np.asarray(raw_image_vector).flatten()
        compressed = self.pca.transform(flat_vector.reshape(1, -1))[0]
        return compressed

    def build_feature_map(self, raw_image_vector, base_symbol="img"):
        self.reset_char()

        features = self.compress_vector(raw_image_vector)
        N = self.k

        input_indices = []
        tensor_arr = []

        current_wires = [self.get_char() for _ in range(N)]
        input_indices.extend([[w] for w in current_wires])
        tensor_arr.extend([(None, '0')] * N)

        for l in range(self.layers):
            op_idx = 0

            # Data injection layer
            for i in range(N):
                nxt = self.get_char()
                input_indices.append(current_wires[i] + nxt)
                tensor_arr.append((features[i], 'Rx'))
                current_wires[i] = nxt

            # Lernable rotation layer
            for i in range(N):
                nxt = self.get_char()
                input_indices.append([current_wires[i], nxt])
                tensor_arr.append((f"{base_symbol}_l{l}_{op_idx}", 'Ry'))
                op_idx += 1

            if N > 1:
                for i in range(N-1):
                    c_idx, t_idx = i, (i + 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    input_indices.append([current_wires[c_idx], current_wires[t_idx], c_out, t_out])
                    tensor_arr.append((f"{base_symbol}_l{l}_{op_idx}", 'CRz'))
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
            
        ccg_map = {0: current_wires}
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr

def modal_compose(text_einsum, text_tensor_arr, img_einsum, img_tensor_arr):
    text_indices = text_einsum[0]
    all_text_chars = {char for tensor in text_indices for char in tensor}
    max_text_char = max(all_text_chars) if all_text_chars else 'a'
    char_offset = ord(max_text_char) + 1

    text_outputs = text_einsum[1]
    img_init_positions = [i for i, tensor in enumerate(img_tensor_arr) if tensor == (None, '0')]
    if len(img_init_positions) != len(text_outputs):
        raise ValueError("Dimension Mismatch between Text outputs and Image inputs!")
    
    img_indices = img_einsum[0]
    boundary_wire_map = {img_indices[pos][0]: text_outputs[i] for i, pos in enumerate(img_init_positions)}

    cleaned_img_indices = []
    cleaned_img_tensor_arr = []
    init_set = set(img_init_positions)

    for i, tensor in enumerate(img_indices):
        if i in init_set:
            continue
        remapped_tensor = []
        for wire_char in tensor:
            if wire_char in boundary_wire_map:
                remapped_tensor.append(boundary_wire_map[wire_char])
            else:
                new_char = "".join(chr(ord(c) + char_offset) for c in wire_char)
                remapped_tensor.append(new_char)
        cleaned_img_indices.append(remapped_tensor)
        cleaned_img_tensor_arr.append(img_tensor_arr[i])
    unified_indices = text_indices + cleaned_img_indices
    unified_tensor_arr = text_tensor_arr + cleaned_img_tensor_arr

    final_outputs = []
    for wire_char in img_indices[-1]:
        if wire_char in boundary_wire_map:
            final_outputs.append(boundary_wire_map[wire_char])
        else:
            new_char = "".join(chr(ord(c) + char_offset) for c in wire_char)
            final_outputs.append(new_char)
    return (unified_indices, final_outputs), unified_tensor_arr

@qml.qnode(dev)
def amplitude_encoding(f=None):
    qml.AmplitudeEmbedding(features=f, wires=range(OUT_DIM), normalize=True)
    return qml.state()

def amplitude_encode_image(image_data):
    amplitude_list = []

    for data in tqdm(image_data):
        amp = amplitude_encoding(data)
        amplitude_list.append(amp)

    return(amplitude_list)

from lambeq.backend.quantum import qubit, Ty, Box
clip_type = Ty().tensor(*[qubit]*OUT_DIM)
def encoding_to_qc(encodings):
    qc_arr = []
    for i, image_encoding in enumerate(encodings):
        clip_shape_train = Box("clip", dom=Ty(), cod=clip_type)
        clip_shape_train.data = image_encoding

        diag_train = clip_shape_train.to_diagram()
        qc_arr.append(diag_train)
    return qc_arr
    

def fs_distance(state1, state2):
    inner_product = torch.sum(state1 * state2.conj(), dim=1)
    return (torch.full(inner_product.size(), torch.pi/2) - torch.acos(inner_product.abs().clamp(0,1)))

def qcosine(bstates1, bstates2, eps=1e-9):
    norm1 = torch.linalg.vector_norm(bstates1, ord=2, dim=1)
    norm2 = torch.linalg.vector_norm(bstates2, ord=2, dim=1)
    inner_product = torch.sum(bstates1 * bstates2.conj(), dim=1)
    return inner_product.abs() / (norm1 * norm2 + eps)

def Rz(phase, dtype=torch.complex64):
    #half_theta = torch.pi * phase

    exp1 = torch.exp(-1j * phase).to(dtype)
    exp2 = torch.exp(1j * phase).to(dtype)

    return torch.tensor([[exp1, 0], [0, exp2]], dtype=dtype)

def BatchRz(phase, dtype=torch.complex64):
    # half_thetas = torch.pi * phase

    exp_neg = torch.exp(-1j * phase).to(dtype)
    exp_pos = torch.exp(1j * phase).to(dtype)

    B = phase.shape[0]
    full_mat = torch.zeros((B, 2, 2), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = exp_neg
    full_mat[:, 1, 1] = exp_pos

    return full_mat

def Rx(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    sin = -1j*torch.sin(phase).to(dtype)
    cos = torch.cos(phase).to(dtype)

    return torch.tensor([[cos, sin], [sin, cos]], dtype=dtype)

def BatchRx(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    cos = torch.cos(phase).to(dtype)
    sin = -1j * torch.sin(phase).to(dtype)

    B = phase.shape[0]
    full_mat = torch.empty((B, 2, 2), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = cos
    full_mat[:, 0, 1] = sin
    full_mat[:, 1, 0] = sin
    full_mat[:, 1, 1] = cos

    return full_mat

def Ry(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    sin = torch.sin(phase).to(dtype)
    cos = torch.cos(phase).to(dtype)

    return torch.tensor([[cos, sin], [-sin, cos]], dtype=dtype)

def BatchRy(phase, dtype=torch.complex64):
    # half_theta = torch.pi * phase

    cos = torch.cos(phase).to(dtype)
    sin = torch.sin(phase).to(dtype)

    B = phase.shape[0]
    full_mat = torch.empty((B, 2, 2), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = cos
    full_mat[:, 0, 1] = sin
    full_mat[:, 1, 0] = -sin
    full_mat[:, 1, 1] = cos

    return full_mat

def CRz(phase):
    return torch.block_diag(torch.eye(2), Rz(phase)).reshape(2,2,2,2)

def BatchCRz(phase, dtype=torch.complex64):
    B = phase.shape[0]

    full_mat = torch.zeros((B, 4, 4), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = 1.0
    full_mat[:, 1, 1] = 1.0
    full_mat[:, 2:, 2:] = BatchRz(phase)

    return full_mat.view(B, 2, 2, 2, 2)

def CRx(phase):
    return torch.block_diag(torch.eye(2), Rx(phase)).reshape(2,2,2,2)

def BatchCRx(phase, dtype=torch.complex64):
    B = phase.shape[0]

    full_mat = torch.zeros((B, 4, 4), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = 1.0
    full_mat[:, 1, 1] = 1.0
    full_mat[:, 2:, 2:] = BatchRx(phase, dtype=dtype)

    return full_mat.view(B, 2, 2, 2, 2)

def CRy(phase):
    return torch.block_diag(torch.eye(2), Ry(phase)).reshape(2,2,2,2)

def BatchCRy(phase, dtype=torch.complex64):
    B = phase.shape[0]

    full_mat = torch.zeros((B, 4, 4), dtype=dtype, device=phase.device)
    full_mat[:, 0, 0] = 1.0
    full_mat[:, 1, 1] = 1.0
    full_mat[:, 2:, 2:] = BatchRy(phase, dtype=dtype)

    return full_mat.view(B, 2, 2, 2, 2)

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
        elif gate == '0_dag':
            wire2q[idx_arr[0]].pop()
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