import numpy as np
import torch
from utils import load_data


def cov(X):
    X = X / np.sqrt(X.size(0) - 1)
    return X.t() @ X


def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).to(dtype)


def eigenvalues(patches):
    n, c, h, w = patches.shape
    Sig = cov(patches.reshape(n, c * h * w))
    V, W = torch.symeig(Sig, eigenvectors=True)
    return V.flip(0), W.t().reshape(c * h * w, c, h, w).flip(0)


def compute_eigenvalues():
    dataset = load_data.get_data(return_sets=True)
    V, W = eigenvalues(
        patches(dataset.train_set["data"][:, :, 4:-4, 4:-4])  # [:10000,:...]
    )  # center crop to remove padding
    return V, W

