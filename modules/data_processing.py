from torch.utils.data import Dataset, Subset
from lambeq import BobcatParser, AtomicType
import random, pickle, torch, os, clip
from modules.grammar_ext import *
from modules.tree2einsum import *
import pennylane as qml
from lambeq.backend.quantum import qubit, Box, Ty, Diagram, Controlled, Sqrt, Rz, Rx, Swap, Symbol
from tqdm import tqdm
import opt_einsum as oe
from itertools import count
import gc
from PIL import Image

clip_model, preprocess = clip.load("ViT-B/32")
parser_path = '/Users/tls/Desktop/Work/COMP0267/assignment_5/COMP0267_CW/bobcat'
if os.path.exists(parser_path):
    ccg_parser = BobcatParser(model_name_or_path=parser_path, cache_dir=parser_path)
else:
    ccg_parser = None
OUT_DIM = 9
clip_type = Ty().tensor(*[qubit]*OUT_DIM)
dev = qml.device('default.qubit', wires=OUT_DIM)

def store_pkl(data, fpathname):
    with open(fpathname, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(fpathname):
    gc.disable()
    try:
        with open(fpathname, 'rb') as f:
            data = pickle.load(f)
    finally:
        gc.enable()
    return data

def subsample_data(dataset, nsamp=0):
    # Using this function without specifying nsamp is equivalent to a random shuffle of the data
    if nsamp == 0: 
        nsamp = len(dataset)
    idx_arr = random.sample(range(0, len(dataset)), nsamp)
    return Subset(dataset, idx_arr)

def collate_fn(batch):
    sentences, pos_images, neg_images = zip(*batch)
    return sentences, pos_images, neg_images

class CLIP_Dataset(Dataset):
    def __init__(self, sentence_circuits, pos_image_circuits, neg_image_circuits):
        self.sentence_circuits = sentence_circuits
        self.pos_image_circuits = pos_image_circuits
        self.neg_image_circuits = neg_image_circuits

    def __len__(self):
        return len(self.sentence_circuits)

    def __getitem__(self, idx):
        sent_circ = self.sentence_circuits[idx]
        pos_img_circ = self.pos_image_circuits[idx]
        neg_img_circ = self.neg_image_circuits[idx]
        return sent_circ, pos_img_circ, neg_img_circ
    
def load_images(fname_arr, fpath):
    img_arr = []
    for img_id in fname_arr:
        img_path = os.path.join(fpath, str(img_id)+'.jpg')
        if os.path.exists(img_path):
            img_arr.append(Image.open(img_path).convert('RGB'))
    return img_arr    

def get_valid_svo(df, fpath):
    drop_idx = [] 
    pos_idx = list(df['pos_image_id'])
    neg_idx = list(df['neg_image_id'])
    
    for dfidx, pidx, nidx in tqdm(zip(range(len(df)), pos_idx, neg_idx)):
        fpos = os.path.join(fpath, f"{pidx}.jpg") 
        fneg = os.path.join(fpath, f"{nidx}.jpg") 
        try:
            if os.path.exists(fpos) and os.path.exists(fneg):
                pass
            else:
                drop_idx.append(dfidx)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return df.drop(df.index[drop_idx]).reset_index(drop=True)
    
def get_valid_aro(df, fpath):
    drop_idx = [] 
    img_idx = list(df['image_id'])
    
    for dfidx, idx in tqdm(zip(range(len(df)), img_idx)):
        img_path = os.path.join(fpath, f"{idx}.jpg") 
        try:
            if os.path.exists(img_path):
                pass
            else:
                drop_idx.append(dfidx)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return df.drop(df.index[drop_idx]).reset_index(drop=True)

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

def get_trees_svo(df, parser=ccg_parser):
    sentence_arr = list(df['corrected_sentence'])
    trees = parser.sentences2trees(sentence_arr, suppress_exceptions=True)
    new_trees = []
    drop_idx = []
    for i, tree in enumerate(trees):
        try:
            if tree is not None:
                new_trees.append(tree._resolved().collapse_noun_phrases())
            else:
                drop_idx.append(i)
        except Exception as e:
            print(f"Error processing tree {i}: {e}")
    df = df.drop(df.index[drop_idx]).reset_index(drop=True)
    df.insert(len(df.columns), 'tree', new_trees)
    return df

def get_trees_aro(df, parser=ccg_parser):
    pos_sent_arr = list(df['true_caption'])
    neg_sent_arr = list(df['false_caption'])
    pos_trees = parser.sentences2trees(pos_sent_arr, suppress_exceptions=True)
    neg_trees = parser.sentences2trees(neg_sent_arr, suppress_exceptions=True)
    new_pos_trees = []
    new_neg_trees = []
    drop_idx = []
    for i, (pos_tree, neg_tree) in enumerate(zip(pos_trees, neg_trees)):
        try:
            if pos_tree is not None and neg_tree is not None:
                new_pos_trees.append(pos_tree._resolved().collapse_noun_phrases())
                new_neg_trees.append(neg_tree._resolved().collapse_noun_phrases())
            else:
                drop_idx.append(i)
        except Exception as e:
            print(f"Error processing tree {i}: {e}")
    df = df.drop(df.index[drop_idx]).reset_index(drop=True)
    df.insert(len(df.columns), 'pos_tree', new_pos_trees)
    df.insert(len(df.columns), 'neg_tree', new_neg_trees)
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

def get_clip_embeddings(images):
    preprocessed = torch.stack([preprocess(img) for img in images])
    with torch.no_grad():
        embeddings = clip_model.encode_image(preprocessed)
    return embeddings.cpu().float()
    
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

def encoding_to_qc(encodings):
    qc_arr = []
    for i, image_encoding in enumerate(encodings):
        clip_shape_train = Box("clip", dom=Ty(), cod=clip_type)
        clip_shape_train.data = image_encoding

        diag_train = clip_shape_train.to_diagram()
        qc_arr.append(diag_train)
    return qc_arr

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