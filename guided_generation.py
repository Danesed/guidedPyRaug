import argparse
import os
import logging
from typing import List

import numpy as np
import torch
import pickle

from pyraug.models.rhvae.rhvae_model import RHVAE
from pyraug.models.rhvae.rhvae_sampler import RHVAESampler
from pyraug.models.rhvae.rhvae_config import RHVAESamplerConfig

from utils.utils import estimate_age, save_generated_volume, plot_tsne

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace) -> None:
    # Check if the latent map pickle exists.
    if not os.path.exists(args.latent_map_pickle):
        logging.error(f"Latent map pickle file not found: {args.latent_map_pickle}")
        return

    with open(args.latent_map_pickle, 'rb') as f:
        latent_map = pickle.load(f)
    latent_vectors = latent_map['latent_vectors']
    age_labels = latent_map['age_labels']
    logging.info(f"Loaded latent map with shape: {latent_vectors.shape}")

    model = RHVAE.load_from_folder(args.path_to_model_folder)
    model.to(device)
    model.eval()
    logging.info("Model loaded and set to eval mode.")

    sampler_config = RHVAESamplerConfig(
        output_dir=args.sampler_output_dir,
        mcmc_steps_nbr=args.mcmc_steps_nbr,
        n_lf=args.n_lf,
        eps_lf=args.eps_lf,
        beta_zero=args.beta_zero,
        batch_size=1,
        samples_per_save=1
    )
    sampler = RHVAESampler(model, sampler_config=sampler_config)

    # Conditional generation based on desired age.
    if args.desired_age is not None:
        candidate_ages: List[float] = []
        candidate_latents: List[torch.Tensor] = []
        logging.info(f"Generating {args.n_candidates} candidate latent samples for desired age: {args.desired_age}")
        for i in range(args.n_candidates):
            with torch.no_grad():
                candidate = sampler.hmc_sampling(1)
            pred_age, _ = estimate_age(candidate, latent_vectors, age_labels)
            candidate_ages.append(pred_age)
            candidate_latents.append(candidate)
            logging.info(f"Candidate {i + 1}: predicted age = {pred_age:.2f}")
        diffs = np.abs(np.array(candidate_ages) - args.desired_age)
        best_idx = int(np.argmin(diffs))
        latent_sample = candidate_latents[best_idx]
        predicted_age = candidate_ages[best_idx]
        logging.info(
            f"Selected candidate {best_idx + 1} with predicted age {predicted_age:.2f} (diff {diffs[best_idx]:.2f}).")
    else:
        with torch.no_grad():
            latent_sample = sampler.hmc_sampling(1)
        predicted_age, _ = estimate_age(latent_sample, latent_vectors, age_labels)
        logging.info(f"Generated latent sample with predicted age: {predicted_age:.2f}")

    with torch.no_grad():
        generated = model.decoder(latent_sample).detach()

    save_generated_volume(generated, predicted_age)

    # Optional t-SNE visualization.
    if args.output_plot is not None:
        plot_tsne(latent_vectors, age_labels, latent_sample, predicted_age, args.output_plot)
    else:
        logging.info("No output_plot provided: skipping t-SNE computation and visualization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an MRI and optionally visualize the latent space via TSNE.")
    parser.add_argument("--path_to_model_folder", type=str, required=True,
                        help="Path to the pretrained model folder.")
    parser.add_argument("--latent_map_pickle", type=str, default="oasis_latents.pkl",
                        help="Path to the latent map pickle file created by create_latent_map.py.")
    parser.add_argument("--sampler_output_dir", type=str, default="outputs",
                        help="Directory to save intermediate sampling outputs.")
    parser.add_argument("--mcmc_steps_nbr", type=int, default=10,
                        help="Number of MCMC steps for HMC sampling.")
    parser.add_argument("--n_lf", type=int, default=1,
                        help="Leapfrog steps factor for HMC sampling.")
    parser.add_argument("--eps_lf", type=float, default=0.1,
                        help="Leapfrog step-size for HMC sampling.")
    parser.add_argument("--beta_zero", type=float, default=1.0,
                        help="Initial beta value for HMC sampling.")
    parser.add_argument("--output_plot", type=str, default="tsne_visualization.png",
                        help="Filename to save the t-SNE plot. If not provided, t-SNE computation is skipped.")
    parser.add_argument("--desired_age", type=float, default=50,
                        help="Desired age for the generated brain. Generation is conditioned on this age if provided.")
    parser.add_argument("--n_candidates", type=int, default=10,
                        help="Number of candidate latent samples to generate when conditioning on a desired age.")
    args = parser.parse_args()
    main(args)