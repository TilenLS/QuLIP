from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import pandas as pid
import numpy as np
import torch, os, clip, pickle, mlflow, time, sys

root_path = os.getcwd() 
os.chdir(root_path)

from modules.data_processing import *
from modules.model import *

NQUBITS = 5
NLAYERS = 3
IMG_DIM = 512

txt_path = os.path.join(root_path, 'ARO/text_circuits/attribution')
img_path = os.path.join(root_path, 'ARO/images')

train_einsum = load_pkl(os.path.join(txt_path, 'aro_train_einsum_as_53.pkl'))
valid_einsum = load_pkl(os.path.join(txt_path, 'aro_valid_einsum_as_53.pkl'))
test_einsum = load_pkl(os.path.join(txt_path, 'aro_test_einsum_as_53.pkl'))
train_pos_einsum, train_neg_einsum = zip(*train_einsum)
valid_pos_einsum, valid_neg_einsum = zip(*valid_einsum)
test_pos_einsum, test_neg_einsum = zip(*test_einsum)

train_img = load_pkl(os.path.join(img_path, 'att_imgenc_train_512.pkl'))
valid_img = load_pkl(os.path.join(img_path, 'att_imgenc_valid_512.pkl'))
test_img = load_pkl(os.path.join(img_path, 'att_imgenc_test_512.pkl'))
arr2type = lambda arr, type: [ele.to(type) for ele in arr]
train_img = arr2type(train_img, torch.complex128)
valid_img = arr2type(valid_img, torch.complex128)
test_img = arr2type(test_img, torch.complex128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

model = EinsumModel()
model.precision = torch.complex128
model.from_einsum(train_pos_einsum + train_neg_einsum + valid_pos_einsum + valid_neg_einsum + test_pos_einsum + test_neg_einsum)

model.initialise_weights(near_id=True, temptrain=False)
model.to(device)
TPARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)

if next(model.parameters()).is_cuda and torch.cuda.is_available():
    print("Model is on GPU")
else:
    print("Running on CPU...")
print(f"Parameters: #{TPARAMS}")

# model.load(os.path.join('aro_runs', '03_05_14:36:29/model.lt')) # attribution
# model.load(os.path.join('aro_runs', '03_05_23:46:35/model.lt')) # relation
# model.load(os.path.join('svo_runs', '03_03_13:28:14/model.lt')) # swap

BATCH_SIZE = 256
N_EXAMPLES = len(train_einsum)
BATCH_SIZE = N_EXAMPLES

generator = torch.Generator(device=device)
train_dataset = CLIP_Dataset(train_pos_einsum, train_neg_einsum, train_img)
train_dataset = subsample_data(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, generator=generator)

valid_dataset = CLIP_Dataset(valid_pos_einsum, valid_neg_einsum, valid_img)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, generator=generator)

test_dataset = CLIP_Dataset(test_pos_einsum, test_neg_einsum, test_img)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, generator=generator)

model.compile_dataset(train_loader, 'aro')
model.compile_dataset(valid_loader, 'aro')
model.compile_dataset(test_loader, 'aro')
print(f"Unique Contractions: {len(model.path_cache)}")

dtime = time.strftime("%m_%d_%H:%M:%S")
fpath = os.path.join(root_path, 'aro_runs', dtime)
if not os.path.exists(fpath):
    os.makedirs(fpath)
db_path = os.path.join(root_path, 'iqp_aro_att.db')
mlflow.pytorch.autolog()
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment('qclip_aro_att')

EPOCHS = 100
LEARNING_RATE = 1e-2
LR_FACTOR = 0.5
MPATIENCE = 2
SEED = int.from_bytes(os.urandom(4))
set_seed(SEED)

optimizer = torch.optim.Adam([{'params': model.weights, 'lr': LEARNING_RATE}])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR, patience=5)
loss_fn = QInfoNCE_cos(device=device)
acc_fn = fs_distance

trajectory = []
initial_params = torch.cat([p.view(-1) for p in model.parameters()]).detach().clone()
print(f"Starting training at: {dtime}")

with mlflow.start_run(run_name=dtime):
    # --- Log Hyperparameters ---
    mlflow.log_param("seed", SEED)
    mlflow.log_param("learning_rate_factor", LR_FACTOR)
    mlflow.log_param("patience", MPATIENCE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("total_parameters", TPARAMS)
    mlflow.log_param("qubits_per_type", NQUBITS)
    mlflow.log_param("ansatze_layers", NLAYERS)


    # --- Start Training ---
    for epoch in trange(EPOCHS):
        # --- Training Pass ---
        train_start = time.time()
        train_loss, train_acc, avg_grad_norm = update_model_aro(model, train_loader, loss_fn, acc_fn, optimizer)
        train_end = time.time()

        # --- Validation Pass ---
        valid_start = time.time()
        valid_acc = eval_model_aro(model, valid_loader, acc_fn)
        valid_end = time.time()

        # --- Record Time & Save Current Weights ---
        model.save(os.path.join(fpath, 'model.lt'))
        train_time = train_end - train_start
        valid_time = valid_end - valid_start

        # --- Compute & Log Relevant Metrics ---
        current_lr = optimizer.param_groups[0]["lr"]
        current_w = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        trajectory.append(current_w)
        init_mad = torch.mean(torch.abs(current_w - initial_params)).item()
        theta_var = torch.mean(torch.abs(current_w - torch.mean(current_w))).item()

        mlflow.log_metric("train_loss", train_loss, step=epoch+1)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch+1)
        mlflow.log_metric("valid_accuracy", valid_acc, step=epoch+1)
        mlflow.log_metric("avg_grad_norm", avg_grad_norm, step=epoch+1)
        mlflow.log_metric("epoch_time", train_time + valid_time, step=epoch+1)
        mlflow.log_metric("learning_rate", current_lr, step=epoch+1)
        mlflow.log_metric("temperature", model.temp, step=epoch+1)
        mlflow.log_metric("parameter_drift", init_mad, step=epoch+1)
        mlflow.log_metric("parameter_variance", theta_var, step=epoch+1)

        # --- Print Current Epoch Information ---
        tqdm.write(f"Epoch {epoch+1}: Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {valid_acc:.4f}, Avg Grad Norm {avg_grad_norm:.2e}, Train Time {train_time/60:.2f}, Valid Time {valid_time/60:.2f} (LR: {current_lr}, T: {model.temp.item():.5f}), Drift: {init_mad:.4f}, Variance: {theta_var:.4f}".strip(), file=sys.stderr)

        if (current_lr <= LEARNING_RATE*LR_FACTOR**MPATIENCE) or (epoch + 1 == EPOCHS):
            mlflow.log_param("epochs", epoch + 1)
            break
        else:
            scheduler.step(valid_acc)

    # --- Evaluate Model on Test Set ---
    test_acc = eval_model_aro(model, test_loader, acc_fn)
    mlflow.log_metric("test_accuracy", test_acc, step=epoch+1)
    print(f"Test Accuracy: {test_acc}")

store_pkl(trajectory, os.path.join(fpath,'trj.pkl'))
mlflow.end_run()