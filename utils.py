import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from torch.utils.data import Subset

def set_seeds(seed=42):
    """
    Sets random seeds for reproducibility across all libraries.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sample_subset_per_class(subset, percentage=0.3, seed=42):
    """
    Samples a specific percentage of data per class to create a balanced subset.

    Args:
        subset (Subset): The original PyTorch Subset object.
        percentage (float): The fraction of data to keep per class (0.0 to 1.0).
        seed (int): Random seed for the sampling process.

    Returns:
        Subset: A new PyTorch Subset containing the sampled indices.
    """
    random.seed(seed)
    dataset = subset.dataset
    indices = subset.indices
    
    # Map dataset indices to their respective classes for stratified sampling
    class_to_indices = defaultdict(list)
    for i in indices:
        label = dataset.targets[i]
        class_to_indices[label].append(i)

    selected_indices = []
    print(f"Sampling {percentage*100}% per class...")
    for cls in class_to_indices:
        cls_indices = class_to_indices[cls]
        k = int(len(cls_indices) * percentage)
        # Randomly sample k indices for the current class
        selected_indices.extend(random.sample(cls_indices, k))

    return Subset(dataset, selected_indices)

def get_optimizer(model, config):
    """
    Creates a PyTorch optimizer based on parameters defined in a config dictionary.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        config (dict): Configuration dictionary containing 'lr' and 'weight_decay'.

    Returns:
        torch.optim.Optimizer: The initialized SGD optimizer.
    """
    lr = config['lr']
    wd = config['weight_decay']
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

def save_training_results(results, file_path):
    """
    Saves training metrics (loss/accuracy) to a JSON file.

    Args:
        results (dict): Dictionary containing lists of training/validation metrics.
        file_path (str): The destination path for the JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def plot_curves(results, save_path):
    """
    Generates and saves loss and accuracy plots from training history.

    Args:
        results (dict): Dictionary containing 'train_loss', 'val_loss', 
            'train_acc', and 'val_acc' history.
        save_path (str): The file path where the resulting plot will be saved.
    """
    epochs = range(len(results["train_loss"]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot Loss (Training vs Validation)
    ax[0].plot(epochs, results["train_loss"], label="Train Loss")
    ax[0].plot(epochs, results["val_loss"], label="Val Loss")
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Plot Accuracy (Training vs Validation)
    ax[1].plot(epochs, results["train_acc"], label="Train Acc")
    ax[1].plot(epochs, results["val_acc"], label="Val Acc")
    ax[1].set_title("Accuracy Curve")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()