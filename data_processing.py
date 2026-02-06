from torch.utils.data import Dataset, Subset
from lambeq import BobcatParser, AtomicType
import random, pickle, torch
from grammar_ext import *
import pennylane as qml
from lambeq.backend.quantum import qubit, Box, Ty
from tqdm import tqdm

ccg_parser = BobcatParser(model_name_or_path='/Users/tls/Desktop/Work/COMP0267/assignment_5/COMP0267_CW/bobcat')
OUT_DIM = 9
clip_type = Ty().tensor(*[qubit]*OUT_DIM)
dev = qml.device('default.qubit', wires=OUT_DIM)

def store_pkl(data, fpathname):
    with open(fpathname, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(fpathname):
    with open(fpathname, 'rb') as f:
        data = pickle.load(f)
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
    
def get_valid_diagrams(df, parser=ccg_parser):
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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/32", device=device)
def get_clip_embeddings(images, model, preprocess, device='cpu'):
    preprocessed = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(preprocessed)
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