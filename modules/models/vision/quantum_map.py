import torch.nn as nn
from itertools import count
import torch
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from opt_einsum import contract_expression
from modules.compilation.quantum.gates import *
from modules.utils.quantum_ops import amplitude_encoding


class FrozenCLIP(nn.Module):
    def __init__(self, clip_model = None, classical=True):
        super().__init__()
        self.clip_model = clip_model
        self.params = nn.ParameterList([])
        self.classical = classical

    def forward(self, img_vecs):
        if self.classical:
            return img_vecs
        else:
            return amplitude_encoding(img_vecs)

class QuantumFeatureMap(nn.Module):
    def __init__(self, k: int, layers: int, batch_size: int, id_init=False):
        super().__init__()
        self.k = k
        self.layers = layers
        self.batch_size = batch_size
        self.params = nn.ParameterList([])
        self.sym2param = {}
        self.pca = IncrementalPCA(n_components=k)

        self.init_params(id_init)
        self.compile_fmap()

    def init_params(self, id_init=False):
        if id_init:
            self.params = nn.Parameter(torch.empty(self.layers * (2*self.k - 1)).uniform_(-0.01, 0.01), requires_grad=True)
        else:
            self.params = nn.Parameter(torch.randn(self.layers * (2*self.k - 1)) * 2 * torch.pi)

    def reset_char(self):
        self.char_idx = count(0)

    def get_char(self):
        i = next(self.char_idx)
        chars = "acdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return chars[i] if i < len(chars) else chr(192 + i - len(chars))

    def compile_fmap(self):
        self.reset_char()
        input_indices = []
        tensor_arr = []
        shape_arr = []

        current_wires = [self.get_char() for _ in range(self.k)]
        input_indices.extend([w for w in current_wires])
        tensor_arr.extend([(None, '0')] * self.k)
        shape_arr.extend([[2]] * self.k)

        symbol_idx = 0
        for l in range(self.layers):
            op_idx = 0
            for i in range(self.k):
                nxt = self.get_char()
                input_indices.append('b' + current_wires[i] + nxt)
                symbol = f"pca_{i}"
                tensor_arr.append((symbol, 'Rx'))
                shape_arr.append([self.batch_size, 2, 2])
                current_wires[i] = nxt

            for i in range(self.k):
                nxt = self.get_char()
                input_indices.append(current_wires[i] + nxt)
                symbol = f"img_l{l}_{op_idx}"
                tensor_arr.append((symbol, 'Ry'))
                self.sym2param[symbol] = symbol_idx
                symbol_idx += 1
                shape_arr.append([2, 2])
                current_wires[i] = nxt
                op_idx += 1

            if self.k > 1:
                for i in range(self.k-1):
                    c_idx, t_idx = i, (i + 1) % self.k
                    c_out, t_out = self.get_char(), self.get_char()
                    input_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    symbol = f"img_l{l}_{op_idx}"
                    tensor_arr.append((symbol, 'CRz'))
                    self.sym2param[symbol] = symbol_idx
                    symbol_idx += 1
                    shape_arr.append([2, 2, 2, 2])
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
        
        einsum_str = f"{','.join(input_indices)}->b{''.join(current_wires)}"
        self.gate_arr = tensor_arr
        self.contraction_path = contract_expression(einsum_str, *shape_arr)

    def fit_image_pca(self, image_stream, batch_size=2048):
        batch = []
        for img_tensor in image_stream:
            batch.append(img_tensor.cpu().numpy() if hasattr(img_tensor, 'numpy') else img_tensor)
            if len(batch) == batch_size:
                self.pca.partial_fit(np.array(batch))
                batch = []
        if batch:
            self.pca.partial_fit(np.array(batch))

    def forward(self, img_vecs):
        if torch.is_tensor(img_vecs):
            flat_vector = img_vecs.detach().cpu().numpy().reshape(len(img_vecs), -1)
        else:
            flat_vector = np.asarray(img_vecs).reshape(len(img_vecs), -1)

        features = self.pca.transform(flat_vector)
        dev = self.params.device
        dtype = self.params.dtype
        tensor_arr = []
        for symbol, gate in self.gate_arr:
            if gate == '0':
                tensor_arr.append(torch.tensor([1, 0], dtype=torch.complex64, device=dev))
            elif gate == 'Rx':
                idx = int(symbol.split('_')[1])
                tensor_arr.append(Rx(torch.tensor(features[:, idx], dtype=dtype, device=dev)))
            elif gate == 'Ry':
                idx = self.sym2param[symbol]
                tensor_arr.append(Ry(self.params[idx]))
            elif gate == 'CRz':
                idx = self.sym2param[symbol]
                tensor_arr.append(CRz(self.params[idx]))
        return self.contraction_path(*tensor_arr)