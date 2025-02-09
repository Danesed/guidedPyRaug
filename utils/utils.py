import numpy as np
import torch
import os
import logging
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from typing import Tuple

def estimate_age(latent_sample: torch.Tensor, latent_vectors: np.ndarray, age_labels: np.ndarray,
                 n_neighbors: int = 5) -> Tuple[float, np.ndarray]:
    """
    Estimate age for a latent sample using k-nearest neighbors.

    Returns:
        predicted_age: median age of the neighbors.
        neighbor_indices: indices of the nearest neighbors.
    """
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn_model.fit(latent_vectors)
    distances, indices = nn_model.kneighbors(latent_sample.cpu().numpy())
    neighbor_ages = age_labels[indices[0]]
    predicted_age = float(np.median(neighbor_ages))
    return predicted_age, indices[0]

def save_generated_volume(generated: torch.Tensor, predicted_age: float) -> str:
    """
    Save the generated volume to a NIfTI file with a filename based on current timestamp and predicted age.

    Returns:
        Full output path of the saved volume.
    """
    generated_vol = generated.cpu().numpy().squeeze().reshape((91, 109, 91))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    volume_filename = f"generated_{timestamp}_age_{predicted_age:.1f}.nii.gz"
    output_volume_path = os.path.join(os.getcwd(), volume_filename)
    gen_img = nib.Nifti1Image(generated_vol, affine=np.eye(4))
    nib.save(gen_img, output_volume_path)
    logging.info(f"Generated volume saved to {output_volume_path}")
    return output_volume_path

def plot_tsne(latent_vectors: np.ndarray, age_labels: np.ndarray, latent_sample: torch.Tensor, predicted_age: float,
              output_plot: str) -> None:
    """
    Compute and save a 2D t-SNE plot of the latent space including the generated sample and its nearest neighbors.
    """
    combined_latents = np.vstack([latent_vectors, latent_sample.cpu().numpy()])
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(combined_latents)
    logging.info("t-SNE computation complete.")

    tsne_latents = latent_tsne[:-1]  # original latent map points.
    tsne_sample = latent_tsne[-1]  # the new sampled latent point.

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne_latents[:, 0], tsne_latents[:, 1], c=age_labels,
                     cmap='viridis', s=30, alpha=0.6, label='Latent Map')
    plt.colorbar(sc, label='Age')
    plt.scatter(tsne_sample[0], tsne_sample[1], marker='*', s=300, c='red',
                label=f'Sampled latent\n(predicted age: {predicted_age:.2f})')

    # Highlight and annotate the 5 nearest neighbors.
    nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_model.fit(latent_vectors)
    _, indices = nn_model.kneighbors(latent_sample.cpu().numpy())
    tsne_neighbors = tsne_latents[indices[0]]
    plt.scatter(tsne_neighbors[:, 0], tsne_neighbors[:, 1],
                marker='o', s=150, facecolors='none', edgecolors='black', linewidths=2,
                label='5 Nearest Neighbors')
    for idx, (x, y) in zip(indices[0], tsne_neighbors):
        plt.annotate(f"{age_labels[idx]:.1f}", (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=10, color='black')

    plt.title('t-SNE of MRI Latent Representations\nwith sampled latent & neighbors')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    logging.info(f"t-SNE plot saved to {output_plot}")
    plt.close()