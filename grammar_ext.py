from lambeq.backend.grammar import Ty, Cup, Word
from collections import deque

def get_symbols(atoms):
    if atoms == Ty():
        return []
    if not atoms.objects:
        return [atoms]
    else:
        return atoms.objects
    
def get_out_idx(target_diag):
    idx_arr = []
    word_syms = []
    for layer in target_diag.layers:
        if type(layer.box) == Cup:
            left = get_symbols(layer.left)
            right = get_symbols(layer.right)
            word_syms = left + right
            idx_arr = idx_arr[:len(left)] + idx_arr[len(left)+2:]
        if type(layer.box) == Word:
            sym_arr = get_symbols(layer.box.cod)
            word_syms += sym_arr
            idx_arr += list(range(len(idx_arr), len(idx_arr)+len(sym_arr)))
    return idx_arr.pop()

def new_sym(oty, nty, idx):
    sym_arr = get_symbols(oty)
    if len(sym_arr) == 1: 
        return nty 
    else: 
        sym_arr[idx] = nty
        return Ty(objects=sym_arr)
    
def replace_out(target_diag):
    output_id = get_out_idx(target_diag)
    special_type = Ty('OUT')
    target_len = output_id+1
    for layer in target_diag.layers:
        #print(output_id, layer)
        left_len = len(get_symbols(layer.left))
        if left_len >= target_len:
            layer.left = new_sym(layer.left, special_type, output_id)
            continue
        box_len = len(get_symbols(layer.box.dom)) + len(get_symbols(layer.box.cod))
        if (left_len + box_len >= target_len) and (type(layer.box) == Word):
            layer.box.cod = new_sym(layer.box.cod, special_type, output_id - left_len)
            continue
        if (left_len + box_len < target_len) and (target_len <= left_len + box_len + len(get_symbols(layer.right))):
            layer.right = new_sym(layer.right, special_type, output_id - left_len - box_len)
        if (type(layer.box) == Cup): 
            output_id -= 2
            target_len -= 2
    target_diag.cod = special_type


def tree2arr(CCG_tree):
    explr_arr = deque([(CCG_tree, 0)])
    ccg_order = []
    left = 0
    
    while True:
        cur_node, left = explr_arr.pop()
        ccg_order.append((cur_node.text, cur_node.rule, left))

        if not cur_node.is_leaf:
            if cur_node.is_binary:
                lchild, rchild = cur_node.children
                explr_arr.appendleft((lchild, left))
                explr_arr.appendleft((rchild, left+len(lchild.text.split(' '))))
            if cur_node.is_unary: 
                explr_arr.appendleft((cur_node.children[0], left))

        if len(explr_arr) == 0:
            break
        left += 1

    return ccg_order

