# Originally from https://github.com/microsoft/otdd (MIT License)
# Modified: added function make_limited_dataset to select subset of data with max samples per class


import torch
from torchvision.models import resnet18, ResNet18_Weights
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance, FeatureCost
from torch.utils.data import Subset, Dataset
import numpy as np

dataset1 = "FashionMNIST"
dataset2 = "CIFAR10"
# If dataset is not fashionMNIST, chamge to3channelse to false

# LOAD DATASETS
loader_1 = load_torchvision_data(
    dataset1,
    resize=32,
    to3channels=True,   # convert grayscale to 3 channels
    maxsize=3000
)[0]


loader_2 = load_torchvision_data(
    dataset2,
    resize=32,
    maxsize=3000
)[0]


# Class to limit data per class to make computing faster
# The make_limited_dataset-function was debugged and developed with the help of ChatGPT
def make_limited_dataset(dataset, max_per_class=10):
    # Returns a Subset of the dataset with at most max_per_class samples per label,

    targets = np.array(dataset.targets)
    indices = []

    for c in np.unique(targets):
        cls_idx = np.where(targets == c)[0]
        indices.extend(cls_idx[:max_per_class])

    subset = Subset(dataset, indices)
    subset.targets = torch.tensor([dataset.targets[i] for i in indices], dtype=torch.long)

    # Wraps the subset to convert 1-channel images to 3 channels if needed.
    class Wrapper(Dataset):
        def __init__(self, subset):
            self.subset = subset
            self.targets = subset.targets
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if x.shape[0] == 1:       # convert grayscale to 3 channels if needed
                x = x.repeat(3,1,1)
            return x, y

    return Wrapper(subset)

# Use subset for faster analysis
train_1_small = make_limited_dataset(loader_1["train"].dataset, max_per_class=10)
train_2_small = make_limited_dataset(loader_2["train"].dataset, max_per_class=10)


# RESNET EMBEDDINGS
# Used as in otdd/advanced_example.py
embedder = resnet18(weights=ResNet18_Weights.DEFAULT)
embedder.fc = torch.nn.Identity()  # remove final classification layer
embedder.eval()
for p in embedder.parameters():
    p.requires_grad = False

feature_cost = FeatureCost(
    src_embedding=embedder,
    src_dim=(3,32,32),
    tgt_embedding=embedder,
    tgt_dim=(3,32,32),
    p=2,
    device="cpu"   # CPU safe
)


# Calculate OTDD DISTANCE
dist = DatasetDistance(
    train_1_small,
    train_2_small,
    feature_cost=feature_cost,
    inner_ot_method="exact",   # exact OT
    debiased_loss=True,
    sqrt_method="spectral",
    sqrt_niters=10,
    p=2,
    device="cpu"
)

# Compute OTDD distance
d = dist.distance(maxsamples=200)
print(f"OTDD distance {dataset1} <-> {dataset2}:", d)