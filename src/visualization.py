import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torch
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
import hydra
import sys
import os
import argparse
from tqdm import tqdm
from omegaconf import DictConfig
from src import utils
from src.models.image_encoder import ImageEncoder
import importlib_metadata as importlib_metadata
if 'importlib.metadata' not in sys.modules:
    sys.modules['importlib.metadata'] = importlib_metadata
from umap import UMAP

DISEASE_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion',
]



log = utils.get_pylogger(__name__)

def get_full_image_path(relative_path, project_dir):
    return os.path.join(project_dir, relative_path)

def extract_features(model, image_paths, transform, project_dir):
    features = []
    # Check if the model is an ImageEncoder (pretrained) or ConVIRT model
    is_pretrained = isinstance(model, ImageEncoder)
    
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting features", unit="image"):
            full_path = get_full_image_path(path, project_dir)
            try:
                img = Image.open(full_path).convert('RGB') 
                img = transform(img).unsqueeze(0)
                
                # Use different model calls based on type
                if is_pretrained:
                    feat = model(img)  # Direct call for ImageEncoder
                else:
                    feat = model.image_model(img)  # Use image_model for ConVIRT
                    
                features.append(feat.cpu().numpy())
            except Exception as e:
                print(f"Error processing {full_path}: {str(e)}")
    return np.vstack(features)

def process_labels(original_labels):
    new_labels = np.zeros((len(original_labels), 7))
    
    disease_map = {
        'No Finding': 0,
        'Atelectasis': 1,
        'Cardiomegaly': 2,
        'Consolidation': 3,
        'Edema': 4,
        'Pleural Effusion': 5
    }
    
    for i in range(len(original_labels)):
        if original_labels[i, 0] == 1:
            new_labels[i, 0] = 1
        else:
            found_selected = False
            for disease, new_idx in disease_map.items():
                old_idx = original_disease_labels.index(disease)
                if original_labels[i, old_idx] == 1:
                    new_labels[i, new_idx] = 1
                    found_selected = True
            
            if not found_selected and original_labels[i, 0] != 1:
                new_labels[i, 6] = 1

    return new_labels

def visualize_embeddings(embedding_results, labels_subset, title, subset_size, visualization_dir):
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8))

    colors = sns.color_palette('husl', n_colors=len(DISEASE_LABELS))

    for i, disease in enumerate(DISEASE_LABELS):
        mask = labels_subset[:, i] == 1
        if mask.any():
            plt.scatter(embedding_results[mask, 0], 
                       embedding_results[mask, 1],
                       c=[colors[i]],
                       label=disease,
                       alpha=0.6,
                       s=50)

    plt.title(f'{title} visualization of CheXpert image features', fontsize=22)
    plt.legend(title='Classes', 
              bbox_to_anchor=(1.05, 1), 
              title_fontsize=20,
              fontsize=16,
              loc='upper left',
              borderaxespad=0.)
    plt.tight_layout()
    
    plt.savefig(os.path.join(visualization_dir, f'{title}-{subset_size}_visualization.png'))
    print(f"Visualization saved to {os.path.join(visualization_dir, f'{title}-{subset_size}_visualization.png')}")
    plt.close()

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="visualization.yaml")
def main(cfg: DictConfig) -> None:
    # Setup paths
    sys.path.append("..")
    # Load data and config
 
    subset_size = cfg.subset_size
    mapping = cfg.mapping
    project_dir = cfg.paths.project_dir
    visualization_dir = cfg.paths.visualization_dir
    data_dir = cfg.paths.data_dir
    
    pkl_path = os.path.join(data_dir, 'chexpert_small.pkl')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = None
    
    if cfg.paths.checkpoint_path is not None:
        checkpoint = torch.load(cfg.paths.checkpoint_path, map_location=torch.device('cpu'))
        print(f"Loaded checkpoint from {cfg.paths.checkpoint_path}")
    if checkpoint is not None and 'state_dict' in checkpoint :
        state_dict = checkpoint['state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                                if k.startswith('image_model') or k.startswith('net.image_model')}
        
        new_state_dict = {k.replace('net.', ''): v for k, v in filtered_state_dict.items()}
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded image model weights!")
        except Exception as e:
            print(f"Error loading state dict: {e}")

    model.eval()

    # Process data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(
            size=cfg.transform[1].size,
            ratio=tuple(cfg.transform[1].ratio)
        ),
        transforms.RandomAffine(
            degrees=cfg.transform[2].degrees,
            translate=tuple(cfg.transform[2].translate),
            scale=tuple(cfg.transform[2].scale)
        ),
        transforms.ColorJitter(
            brightness=tuple(cfg.transform[3].brightness), 
            contrast=tuple(cfg.transform[3].contrast) 
        ),
        transforms.GaussianBlur(
            kernel_size=cfg.transform[4].kernel_size,
            sigma=tuple(cfg.transform[4].sigma) 
        ),
        transforms.RandomHorizontalFlip(p=cfg.transform[5].p),
        transforms.Resize(cfg.transform[6].size),
        transforms.Normalize(
            mean=cfg.transform[7].mean,
            std=cfg.transform[7].std
        )
    ])

    print(f"Transform: {transform}")


    subset_size = subset_size if subset_size is not None else len(data)
    print(f"Processing subset of {subset_size} samples")

    # image_paths = data['Path'].values[:subset_size]
    global original_disease_labels
    original_disease_labels = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    if subset_size is not None and subset_size < len(data):
        random_indices = np.random.choice(len(data), size=subset_size, replace=False)
        image_paths = data['Path'].values[random_indices]
        # Make sure to use the same indices for labels
        labels_subset = process_labels(data[original_disease_labels].iloc[random_indices].values)
    else:
        image_paths = data['Path'].values
        labels_subset = process_labels(data[original_disease_labels].values)

    features = extract_features(model, image_paths, transform, project_dir)
    print(f"Extracted features shape: {features.shape}")

    # Process labels
    # global original_disease_labels
    # original_disease_labels = [
    #     'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    #     'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    #     'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    #     'Pleural Other', 'Fracture', 'Support Devices'
    # ]
    # labels_subset = process_labels(data[original_disease_labels].iloc[:subset_size].values)

    # Perform dimensionality reduction
    print(f"Performing dimensionality reduction using {mapping}")
    if mapping == 'tsne':
        tsne_params = cfg.get("tsne")
        print(f"t-SNE parameters: {tsne_params}")
        reducer = TSNE(n_components=tsne_params.n_components, 
                      random_state=tsne_params.random_state, 
                      perplexity=tsne_params.perplexity, 
                      n_iter=tsne_params.n_iter, 
                      learning_rate=tsne_params.learning_rate)
        embedding_results = reducer.fit_transform(features)
        method_name = 't-SNE'
    else:  # umap
        umap_params = cfg.get("umap")
        print(f"UMAP parameters: {umap_params}")
        reducer = UMAP(n_components=umap_params.n_components, 
                      n_neighbors=umap_params.n_neighbors, 
                      min_dist=umap_params.min_dist, 
                      random_state=umap_params.random_state)
        embedding_results = reducer.fit_transform(features)
        method_name = 'UMAP'

    print("Visualizing results...")
    visualize_embeddings(embedding_results, labels_subset, method_name, subset_size, visualization_dir)
    print("Visualization complete!")

if __name__ == "__main__":
    main()