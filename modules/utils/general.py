import torch, pickle, gc, random
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            _ = test_tensor + 1
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"\n[Warning] CUDA is physically available but incompatible or broken.")
            print(f"Error detail: {e}")
            print("Falling back to CPU execution for stability...\n")
    return torch.device("cpu")

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

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
