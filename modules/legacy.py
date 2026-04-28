from lambeq import BobcatParser, AtomicType
from lambeq.backend.quantum import Diagram, Sqrt, Swap, Symbol
import os, clip
import opt_einsum as oe
from modules.grammar import *
from modules.quantum import *

clip_model, preprocess = clip.load("ViT-B/32")
parser_path = '/Users/tls/Desktop/Work/COMP0267/assignment_5/COMP0267_CW/bobcat'
if os.path.exists(parser_path):
    ccg_parser = BobcatParser(model_name_or_path=parser_path, cache_dir=parser_path)
else:
    ccg_parser = None

def get_diags_svo(df, parser=ccg_parser):
    sentence_arr = list(df['corrected_sentence'])
    diagrams = parser.sentences2diagrams(sentence_arr)
    new_diagrams = []
    drop_idx = []

    for i, diag in enumerate(diagrams): 
        try:
            if diag.cod == AtomicType.SENTENCE:
                replace_out(diag)
                new_diagrams.append(diag)
            else:
                drop_idx.append(i)
        except:
            print(i)
    df = df.drop(df.index[drop_idx]).reset_index(drop=True)
    df.insert(len(df.columns), 'diagram', new_diagrams)
    return df

def get_diags_aro(df, parser=ccg_parser):
    pos_sent_arr = list(df['true_caption'])
    neg_sent_arr = list(df['false_caption'])
    pos_diags = parser.sentences2diagrams(pos_sent_arr, suppress_exceptions=True)
    neg_diags = parser.sentences2diagrams(neg_sent_arr, suppress_exceptions=True)
    new_pos_diags = []
    new_neg_diags = []
    drop_idx = []
    for i, (pos_diag, neg_diag) in enumerate(zip(pos_diags, neg_diags)):
        try:
            if pos_diag is not None and neg_diag is not None:
                replace_out(pos_diag)
                replace_out(neg_diag)
                new_pos_diags.append(pos_diag)
                new_neg_diags.append(neg_diag)
            else:
                drop_idx.append(i)
        except:
            print(i)
    df = df.drop(df.index[drop_idx]).reset_index(drop=True)
    df.insert(len(df.columns), 'pos_diagram', new_pos_diags)
    df.insert(len(df.columns), 'neg_diagram', new_neg_diags)
    return df

def elim_CNOT(circuit):
    new_layers = []
    for layer in circuit.layers:
        left, box, right = layer.unpack()
        if hasattr(box, 'decompose'):
            sub_circuit = box.decompose()
            for sublayer in sub_circuit.layers:
                l, b, r = sublayer.unpack()
                new_layers.append(circuit.category.Layer(left @ l, b, r @ right))
        else:
            new_layers.append(layer)
    return Diagram(circuit.dom, circuit.cod, new_layers)


def qc_to_einsum(circ):
    idx_gen = count(0)
    def get_new_index():
        new_index = next(idx_gen)
        return new_index
    
    tensors = []
    tensor_edges = [] 
    qubits = []

    for i, layer in enumerate(circ.layers):
        l, box, _ = layer.unpack()
        pos = len(l)
        if isinstance(box, Sqrt):
            tensors.append((None, 'sqrt'))
            tensor_edges.append([])
            continue
        if isinstance(box, Swap):
            qubits[pos], qubits[pos+1] = qubits[pos+1], qubits[pos]
            continue 
        
        n_in = len(box.dom)

        in_edges = qubits[pos : pos + n_in]

        out_edges = [get_new_index() for _ in box.cod.dim]

        qubits = qubits[:pos] + out_edges + qubits[pos + n_in:]

        if isinstance(box.data, Symbol):
            tensors.append((box.data.name, box.name.split('(')[0]))
        else:
            tensors.append((None, box.name))
        tensor_edges.append(in_edges + out_edges)

    subs = [''.join(oe.get_symbol(i) for i in indices) for indices in tensor_edges]
    output_subs = ''.join(oe.get_symbol(i) for i in qubits)
    einsum_string = ','.join(subs) + '->' + output_subs
    
    return einsum_string, tensors

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