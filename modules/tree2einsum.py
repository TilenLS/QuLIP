from lambeq.text2diagram import CCGType, CCGRule 
from itertools import count
from collections import Counter
from abc import ABC, abstractmethod
from tqdm import tqdm, trange   


def curry_type(ccg_type):
    if ccg_type.is_atomic:
        return ccg_type
    
    res = curry_type(ccg_type.result)
    arg = curry_type(ccg_type.argument)     
    
    if res.is_over and ccg_type.is_over:
        return CCGType(result=CCGType(result=res.result, 
                                      direction=ccg_type.direction, 
                                      argument=arg),
                       direction=res.direction,
                       argument=res.argument)
    
    return CCGType(result=res, direction=ccg_type.direction, argument=arg)

def apply_curry_to_tree(node):
    node.biclosed_type = curry_type(node.biclosed_type)
    if not node.is_leaf:
        for child in node.children:
            apply_curry_to_tree(child)
    return node

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
    # root_node = root_node._resolved().collapse_noun_phrases()
    idx_gen = count(0)
    def get_new_index():
        new_index = next(idx_gen)
        return new_index
    
    root_idx = get_new_index()
    stack = [(root_node, [root_idx])]
    tn = []

    while stack: 
        node, idx_arr = stack.pop()
        if node.rule == CCGRule.LEXICAL:
            # tn[node.text] = (idx_arr, get_type(node.biclosed_type))
            tn.append((node.text, idx_arr, get_type(node.biclosed_type)))
        elif node.rule == CCGRule.FORWARD_APPLICATION:
            if node.left.biclosed_type.to_string() in ['(NP/N)', '(n/n)']:
                stack.append((node.right, idx_arr))
                continue
            shared_idx = [get_new_index() for _ in get_type(node.right.biclosed_type)]
            stack.append((node.right, shared_idx))
            stack.append((node.left, idx_arr + shared_idx[::-1]))
        elif node.rule == CCGRule.BACKWARD_APPLICATION:    
            shared_idx = [get_new_index() for _ in get_type(node.left.biclosed_type)]
            stack.append((node.right, shared_idx[::-1] + idx_arr))
            stack.append((node.left, shared_idx))
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
            stack.append((node.right, idx_arr))
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            stack.append((node.left, idx_arr))

    return tn

def cgg_flag_output(tn):
    for i, (w, idx_arr, sym_arr) in enumerate(tn):
        if 0 in idx_arr: 
            j = idx_arr.index(0)
            sym_arr[j] = 'out'
            tn[i] = (w, idx_arr, sym_arr)

def tree2circ_aro(df, ansatze, curry=False):
    einsum_arr = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        pos_tn = tree2einsum(row['pos_tree'])
        neg_tn = tree2einsum(row['neg_tree'])
        cgg_flag_output(pos_tn)
        cgg_flag_output(neg_tn)
        pos_einsum = ansatze(pos_tn, curry=curry)
        neg_einsum = ansatze(neg_tn, curry=curry)
        pos_output_len = len(pos_einsum[0].split('->')[1])
        neg_output_len = len(neg_einsum[0].split('->')[1])
        if pos_output_len == 9 and neg_output_len == 9:
            einsum_arr.append((pos_einsum, neg_einsum))
    return einsum_arr

def tree2circ_svo(df, ansatze, curry=False):
    einsum_arr = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        tn = tree2einsum(row['tree']._resolved().collapse_noun_phrases())
        cgg_flag_output(tn)
        einsum = ansatze(tn, curry=curry)
        einsum_arr.append(einsum)
    return einsum_arr

class BaseAnsatz(ABC):
    def __init__(self, obmap=set(), layers=1):
        super().__init__()
        self.obmap = obmap
        self.layers = layers
        self.char_idx = count(0)

    def __call__(self, tn, curry=False):
        if curry:
            return self.tn2ansatz(tn)
        else:
            return self.tn2ansatz_v2(tn)
        
    def _convert(self, tn):
        return self._tn2ansatz(tn)

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


    def _gen_einsum_expr(self, input_indices, ccg_map):
        idx_counts = Counter("".join(input_indices))
        output_indices = [c for c, cnt in idx_counts.items() if cnt == 1]
        
        # Prioritize the root index (0) wires in the output string
        root_indices = ccg_map.get(0, [])
        out_str = "".join(c for c in root_indices if c in output_indices) + "".join(c for c in output_indices if c not in root_indices)
        return ",".join(input_indices) + "->" + out_str
    

    def gen_einsum_expr(self, input_indices, ccg_map):
        all_idx = [i for sub in input_indices for i in sub]
        idx_counts = Counter(all_idx)
        output_indices = [idx for idx, cnt in idx_counts.items() if cnt == 1]

        root_indices = ccg_map.get(0, ccg_map.get('0', []))
        out_list = [i for i in root_indices if i in output_indices] + \
                   [i for i in output_indices if i not in root_indices]
        
        # (input_subscripts, output_subscript)
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
                input_indices.append(w)
                tensor_arr.append((None, '0'))

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            replace_map = dict(zip(current_wires, out_wires))
            input_indices = ["".join(replace_map.get(c, c) for c in inp) for inp in input_indices] 
        
        einsum_expr = self._gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr
    
    def tn2ansatz(self, tn):
        self.reset_char()
        ccg_map = self.ccg_map(tn)
        input_indices, tensor_arr = [], []

        lifespans = Counter()
        for _, idx_arr, _ in tn:
            for idx in idx_arr:
                for gw in ccg_map[idx]:
                    lifespans[gw] += 1
        external_wires = {gw for gw, count in lifespans.items() if count == 1}
        free_qubits = []
        active_wires = {}
        cutwidth_counter = 0

        for word, idx_arr, type_arr in tn:
            global_wires = sum([ccg_map[idx] for idx in idx_arr], [])
            ansatz_inputs = []

            for gw in global_wires:
                if gw not in active_wires:
                    if free_qubits:
                        current_leg = free_qubits.pop()
                    else: 
                        current_leg = self.get_char()
                        cutwidth_counter += 1
                        input_indices.append([current_leg])
                        tensor_arr.append((None, '0'))
                    active_wires[gw] = current_leg
                ansatz_inputs.append(active_wires[gw])

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires_out, new_indices, new_tensors = self.ansatz(ansatz_inputs, base_symbol)

            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)
            for gw, out_leg in zip(global_wires, current_wires_out):
                active_wires[gw] = out_leg
                lifespans[gw] -= 1
                if lifespans[gw] == 0 and gw not in external_wires:
                    free_qubits.append(out_leg)
                    del active_wires[gw]

        final_replace_map = {leg: gw for gw, leg in active_wires.items()}
        input_indices = [[final_replace_map.get(w, w) for w in sub] for sub in input_indices]
        einsum_obj = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_obj, tensor_arr

    def tn2ansatz_v2(self, tn):
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
            input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]
        
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