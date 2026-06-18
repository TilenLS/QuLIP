from torch.utils.data import Dataset, Subset
from lambeq import BobcatParser
import random, pickle, torch, os, clip
from modules.grammar import *
from modules.quantum import *
from tqdm import tqdm
import gc
from PIL import Image
import pandas as pd

clip_model, preprocess = clip.load("ViT-B/32")
parser_path = '/Users/tls/Desktop/Work/COMP0267/assignment_5/COMP0267_CW/bobcat'
ccg_parser = BobcatParser(model_name_or_path=parser_path, cache_dir=parser_path)

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

def get_valid_images(df, fpath, img_labels='image_id'):
    drop_idx = [] 

    for idx in range(len(df)):
        img_idx_arr = df[img_labels].iloc[idx]
        for img_idx in img_idx_arr:
            img_path = os.path.join(fpath, str(img_idx)+'.jpg')
            try:
                if os.path.exists(img_path):
                    pass
                else:
                    drop_idx.append(idx)
                    break
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return df.drop(df.index[drop_idx]).reset_index(drop=True)

import numpy as np
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_clip_embeddings(images):
    preprocessed = torch.stack([preprocess(img) for img in images])
    with torch.no_grad():
        embeddings = clip_model.encode_image(preprocessed)
    return embeddings.cpu().float()


# def coco_stream2df(data_subset):
#     data_list = []
    
#     for entry in data_subset:
#         data_list.append({
#             'cocoid': entry['cocoid'],
#             'filename': entry['filename'],
#             'caption': entry['caption']
#         })
    
#     df = pd.DataFrame(data_list)
#     return df

# def get_valid_svo(df, fpath):
#     drop_idx = [] 
#     pos_idx = list(df['pos_image_id'])
#     neg_idx = list(df['neg_image_id'])
    
#     for dfidx, pidx, nidx in tqdm(zip(range(len(df)), pos_idx, neg_idx)):
#         fpos = os.path.join(fpath, f"{pidx}.jpg") 
#         fneg = os.path.join(fpath, f"{nidx}.jpg") 
#         try:
#             if os.path.exists(fpos) and os.path.exists(fneg):
#                 pass
#             else:
#                 drop_idx.append(dfidx)
#         except Exception as e:
#             print(f"Error loading {fpath}: {e}")

#     return df.drop(df.index[drop_idx]).reset_index(drop=True)
    
# def get_valid_aro(df, fpath):
#     drop_idx = [] 
#     img_idx = list(df['image_id'])
    
#     for dfidx, idx in tqdm(zip(range(len(df)), img_idx)):
#         img_path = os.path.join(fpath, f"{idx}.jpg") 
#         try:
#             if os.path.exists(img_path):
#                 pass
#             else:
#                 drop_idx.append(dfidx)
#         except Exception as e:
#             print(f"Error loading {img_path}: {e}")

#     return df.drop(df.index[drop_idx]).reset_index(drop=True)