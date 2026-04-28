import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from sklearn.decomposition import PCA
from tqdm import trange, tqdm
from jax.flatten_util import ravel_pytree
import numpy as np
import plotly.graph_objects as go
import os


def plot_fidelity_dist(txt, img, epoch=0, log_scale=False):
    B = txt.size(0)
    sim_matrix = (txt @ img.conj().t()).abs().cpu().detach()

    positives = torch.diagonal(sim_matrix).numpy()
    mask = ~torch.eye(B, dtype=torch.bool)
    negatives = sim_matrix[mask].numpy()

    plt.figure(figsize=(10, 6))
    # Use log_scale=True to see small differences near zero
    sns.kdeplot(positives, label='Positives', fill=True, color='green', log_scale=log_scale)
    sns.kdeplot(negatives, label='Negatives', fill=True, color='red', log_scale=log_scale)

    plt.title(f"Log-Scaled Fidelity Distribution - Epoch {epoch}")
    plt.xlabel("Fidelity (Log Scale)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_sim_matrix(txt, img, epoch=0):
    # Compute similarity matrix
    sims = (txt @ img.conj().t()).abs().cpu().detach().numpy()
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(sims, annot=False, cmap='magma', vmin=0, vmax=1)
    plt.title(f"Fidelity Matrix - Epoch {epoch}")
    plt.xlabel("Image Index")
    plt.ylabel("Text Index")
    plt.savefig(f"sim_matrix_epoch_{epoch}.png")

def plot_training_phase(param_drift_arr, grad_norm_arr):
    # history is a dict with lists: history['drift'], history['grad_norm']
    epochs = range(len(param_drift_arr))

    plt.figure(figsize=(8, 6))
    plt.scatter(grad_norm_arr, param_drift_arr, c=epochs, cmap='viridis')
    plt.colorbar(label='Epoch')
    plt.xlabel("Average Gradient Norm")
    plt.ylabel("Weight Drift (radians)")
    plt.title("Training Phase Portrait")
    
    # Annotate start and end
    plt.annotate('Start', (grad_norm_arr[0], param_drift_arr[0]))
    plt.annotate('End', (grad_norm_arr[-1], param_drift_arr[-1]))
    
    plt.grid(alpha=0.3)
    plt.savefig("phase_portrait.png")

def plot_tsne_alignment(txt, img, epoch=0):
    B = txt.size(0)
    # Concatenate real and imag parts: (B, 2*D)
    txt_flat = torch.cat([txt.real, txt.imag], dim=-1).cpu().detach().numpy()
    img_flat = torch.cat([img.real, img.imag], dim=-1).cpu().detach().numpy()
    
    combined = np.vstack([txt_flat, img_flat])
    tsne = TSNE(n_components=2, perplexity=min(30, B-1))
    proj = tsne.fit_transform(combined)
    
    txt_proj = proj[:B]
    img_proj = proj[B:]

    plt.figure(figsize=(8, 8))
    plt.scatter(txt_proj[:, 0], txt_proj[:, 1], c='blue', label='Text', alpha=0.6)
    plt.scatter(img_proj[:, 0], img_proj[:, 1], c='orange', label='Image', alpha=0.6)
    
    # Draw lines connecting matches
    for i in range(B):
        plt.plot([txt_proj[i, 0], img_proj[i, 0]], 
                 [txt_proj[i, 1], img_proj[i, 1]], 'gray', alpha=0.2, lw=0.5)
                 
    plt.title(f"t-SNE Modality Alignment - Epoch {epoch}")
    plt.legend()
    plt.savefig(f"tsne_epoch_{epoch}.png")


def generate_landscape(model, trajectory, resolution=25, ret_path=False, padding=0.25, loss_offset=0.05):
    _, unflatten_fn = ravel_pytree(model.params)

    traj_np = np.array(trajectory)
    final_state = traj_np[-1].flatten()
    centered_traj = traj_np - unflatten_fn(final_state)
    centered_traj = np.array([w.flatten() for w in centered_traj])

    pca = PCA(n_components=2)
    pca.fit(centered_traj)
    v1 = pca.components_[0].flatten()
    v2 = pca.components_[1].flatten()

    proj_x = centered_traj @ v1
    proj_y = centered_traj @ v2

    x_margin = (proj_x.max() - proj_x.min()) * padding
    y_margin = (proj_y.max() - proj_y.min()) * padding

    X = np.linspace(proj_x.min() - x_margin, proj_x.max() + x_margin, resolution)
    Y = np.linspace(proj_y.min() - y_margin, proj_y.max() + y_margin, resolution)

    Z = np.zeros((resolution, resolution))

    print("Calculating Loss Landscape Surface...")
    with tqdm(total=resolution*resolution) as pbar:
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                new_weights = final_state + (x * v1) + (y * v2)
                Z[i, j] = model.cost_fn(unflatten_fn(new_weights))
                pbar.update(1)

    if ret_path:
        print("Projecting Trajectory...")
        projected_losses = []
        proj_traj = np.stack([proj_x, proj_y], axis=1)
        for i in trange(len(proj_traj)):
            projected_weights = final_state + (proj_x[i] * v1) + (proj_y[i] * v2)
            projected_losses.append(model.cost_fn(unflatten_fn(projected_weights)))
        
        epsilon = (max(projected_losses) - min(projected_losses)) * loss_offset
        loss_path = [z + epsilon for z in projected_losses]
        return (X, Y, Z.T), (proj_traj[:, 0], proj_traj[:, 1], loss_path)

    return (X, Y, Z.T)


def visualize_landscape(landscape, loss_path=None, save_path=None):
    fig = go.Figure()

    X, Y, Z = landscape
    z_min = np.min(Z)
    z_max = np.max(Z)
    z_range = z_max - z_min
    step_size = z_range / 30 

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='RdBu',
        reversescale=True,
        opacity=0.8,
        name='Loss Surface',
        showscale=True,
        colorbar=dict(title="Loss", x=-0.1),
        contours_z=dict(show=True, 
                        usecolormap=True, 
                        highlightcolor="white", 
                        project_z=True, width=4, 
                        size=step_size, 
                        highlightwidth=6, 
                        start=z_min, 
                        end=z_max)
    ))

    if loss_path is not None:
        x, y, z = loss_path
        steps = list(range(len(x)))
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines+markers',
            marker=dict(
                size=5,
                color=steps,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Training Step", x=1.1),
                line=dict(color='white', width=1)
            ),
            line=dict(
                color='black',
                width=6
            ),
            name='Training Path'
        ))

    fig.update_layout(
        title='Parameter Loss Landscape',
        template="plotly_dark",
        scene=dict(
            xaxis_title='PC 1',
            yaxis_title='PC 2',
            zaxis_title='Loss',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8),
            xaxis=dict(backgroundcolor="rgb(20, 24, 33)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(backgroundcolor="rgb(20, 24, 33)", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(backgroundcolor="rgb(20, 24, 33)", gridcolor="rgba(255,255,255,0.1)", zeroline=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=900, height=700
    )

    fig.show()
    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.write_html(os.path.join(save_path, f"loss_landscape_{timestamp}.html"))
