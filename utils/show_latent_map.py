import argparse
import os
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

logging.basicConfig(level=logging.INFO)


def main(args):
    # Load the latent map pickle file.
    with open(args.pickle_file, 'rb') as f:
        latent_map = pickle.load(f)
    latent_vectors = latent_map['latent_vectors']
    age_labels = latent_map['age_labels']
    logging.info(f"Loaded latent map with shape: {latent_vectors.shape}")

    # Run t-SNE on the latent representations.
    tsne = TSNE(n_components=3, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)
    logging.info("t-SNE 3D computation complete.")

    # Create a 3D scatter plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], latent_tsne[:, 2],
                    c=age_labels, cmap='viridis', s=50, alpha=0.7)
    fig.colorbar(sc, ax=ax, label='Age')
    ax.set_title('3D t-SNE of MRI Latent Representations')
    ax.set_xlabel('t-SNE component 1')
    ax.set_ylabel('t-SNE component 2')
    ax.set_zlabel('t-SNE component 3')
    plt.tight_layout()

    # Show or save the plot.
    if args.output_plot:
        plt.savefig(args.output_plot, dpi=300)
        logging.info(f"Plot saved to {args.output_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the latent space via 3D t-SNE. Points are colored according to age."
    )
    parser.add_argument("--pickle_file", type=str, required=False,
                        default="../oasis_latents.pkl",
                        help="Path to the latent map pickle file.")
    parser.add_argument("--output_plot", type=str, required=False,
                        help="Filename to save the plot (e.g., 'latent_tsne_3d.png'). If not provided, the plot is shown interactively.")
    args = parser.parse_args()
    main(args)