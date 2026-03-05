from lambeq.text2diagram import CCGType, CCGRule 
from itertools import count
from collections import Counter
from abc import ABC, abstractmethod

def get_type(ccgtype):
    res_arr = []
    type_arr = [ccgtype]
    while type_arr: 
        cur_type = type_arr.pop()
        if cur_type.is_over:
            type_arr.append(cur_type.result)
            if cur_type.argument.is_complex: 
                old_arg = cur_type.argument
                new_arg = CCGType(result=old_arg.argument, direction=old_arg.direction, argument=old_arg.result)
                type_arr.append(new_arg)
            else:
                type_arr.append(cur_type.argument)
        elif cur_type.is_under: 
            if cur_type.argument.is_complex:
                old_arg = cur_type.argument
                new_arg = CCGType(result=old_arg.argument, direction=old_arg.direction, argument=old_arg.result)
                type_arr.append(new_arg)
            else:
                type_arr.append(cur_type.argument)
            type_arr.append(cur_type.result)
        else: 
            res_arr.append(cur_type.name)
    return res_arr[::-1]


def tree2einsum(root_node):
    root_node = root_node._resolved().collapse_noun_phrases()
    idx_gen = count(0)
    def get_new_index():
        new_index = next(idx_gen)
        return new_index
    
    root_idx = get_new_index()
    stack = [(root_node, [root_idx])]
    # tn = {}
    tn = []

    while stack: 
        node, idx_arr = stack.pop()
        if node.rule == CCGRule.LEXICAL:
            # tn[node.text] = (idx_arr, get_type(node.biclosed_type))
            tn.append((node.text, idx_arr, get_type(node.biclosed_type)))
        elif node.rule == CCGRule.FORWARD_APPLICATION:
            shared_idx = [get_new_index() for _ in get_type(node.right.biclosed_type)]
            stack.append((node.right, shared_idx))
            stack.append((node.left, idx_arr + shared_idx[::-1]))
        elif node.rule == CCGRule.BACKWARD_APPLICATION:    
            shared_idx = [get_new_index() for _ in get_type(node.left.biclosed_type)]
            stack.append((node.right, shared_idx[::-1] + idx_arr))
            stack.append((node.left, shared_idx))
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
            stack.append(node.right, idx_arr)
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            stack.append(node.left, idx_arr)

    return tn

def cgg_flag_output(tn):
    for i, (w, idx_arr, sym_arr) in enumerate(tn):
        if 0 in idx_arr: 
            j = idx_arr.index(0)
            sym_arr[j] = 'out'
            tn[i] = (w, idx_arr, sym_arr)

class BaseAnsatz(ABC):
    def __init__(self, obmap=set(), layers=1):
        super().__init__()
        self.obmap = obmap
        self.layers = layers
        self.char_idx = count(0)

    def __call__(self, tn):
        return self.tn2ansatz(tn)

    def tns2ansatze(self, tn_arr):
        return [self.tn2ansatze(tn) for tn in tn_arr]

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


    def gen_einsum_expr(self, input_indices, ccg_map):
        idx_counts = Counter("".join(input_indices))
        output_indices = [c for c, cnt in idx_counts.items() if cnt == 1]
        
        # Prioritize the root index (0) wires in the output string
        root_indices = ccg_map.get(0, [])
        out_str = "".join(c for c in root_indices if c in output_indices) + "".join(c for c in output_indices if c not in root_indices)
        return ",".join(input_indices) + "->" + out_str


    def tn2ansatz(self, tn):
        self.reset_char()
        ccg_map = self.ccg_map(tn)
        input_indices, tensor_arr = [], []

        for word, idx_arr, type_arr in tn:
            out_wires = sum([ccg_map[idx] for idx in idx_arr], [])
            N = len(out_wires)
            current_wires = [self.get_char() for _ in range(N)]

            for w in current_wires:
                input_indices.append(w)
                tensor_arr.append((None, '0'))

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            replace_map = dict(zip(current_wires, out_wires))
            input_indices = ["".join(replace_map.get(c, c) for c in inp) for inp in input_indices] 
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr

    @abstractmethod
    def ansatz(self, current_wires, base_symbol):
        pass
        

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