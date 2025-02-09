import argparse
import os
import logging
import pickle
import numpy as np
import torch
import nibabel as nib
import pandas as pd

from pyraug.models.rhvae.rhvae_model import RHVAE

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_volume(volume_path):
    img = nib.load(volume_path)
    volume = img.get_fdata()
    # Normalize and add channel dimension (model expects shape (1, D, H, W))
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume = np.expand_dims(volume, axis=0)
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).to(device)
    return volume_tensor


def extract_subject_id(filename):
    basename = os.path.basename(filename)
    if basename.startswith("sub-"):
        basename = basename[len("sub-"):]
    subject_id = basename.split("_")[0]
    return subject_id


def main(args):
    labels_df = pd.read_csv(args.csv_file)
    # Load the pretrained model
    model = RHVAE.load_from_folder(args.path_to_model_folder)
    model.to(device)
    model.eval()
    logging.info("Model loaded and set to eval mode.")

    latent_list = []
    age_list = []

    # Accept multiple volumes folders.
    for folder in args.volumes_folders:
        if not os.path.exists(folder):
            logging.warning(f"Folder {folder} does not exist, skipping.")
            continue
        volume_files = [os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.endswith('.nii.gz')]
        logging.info(f"Found {len(volume_files)} files in {folder}.")
        for vol_path in volume_files:
            subject_id = extract_subject_id(vol_path)
            matches = labels_df[labels_df['list1_id'].str.contains(subject_id, na=False)]
            if matches.empty:
                logging.warning(f"Subject ID {subject_id} not found in CSV, skipping volume {vol_path}.")
                continue
            age = float(matches.iloc[0]['ageAtVisit'])
            vol_tensor = load_volume(vol_path)
            with torch.no_grad():
                mu, _ = model.encoder(vol_tensor)
            latent = mu.cpu().numpy().squeeze()
            latent_list.append(latent)
            age_list.append(age)

    if len(latent_list) == 0:
        logging.error("No volumes processed!")
        return

    latent_vectors = np.vstack(latent_list)
    age_labels = np.array(age_list)
    logging.info(f"Encoded latent representations shape: {latent_vectors.shape}")

    latent_map = {'latent_vectors': latent_vectors, 'age_labels': age_labels}
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(latent_map, f)
    logging.info(f"Latent map saved to {args.output_pickle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model_folder", type=str, required=True,
                        help="Path to the pretrained model folder.")
    parser.add_argument("--volumes_folders", type=str, nargs='+', required=True,
                        help="Space-separated paths to folders with input MRI volumes (nii.gz files).")
    parser.add_argument("--csv_file", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "final_labels_oasis-1.csv"),
                        help="CSV file mapping volume IDs to AgeAtVisit.")
    parser.add_argument("--output_pickle", type=str, default="oasis_latents.pkl",
                        help="Filename for saving the latent map (pickle format).")
    args = parser.parse_args()
    main(args)