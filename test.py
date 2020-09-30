import torch

from models.flow_classifier import SdeClassifier

if __name__ == "__main__":
    classifier = SdeClassifier(n_scale=3, nclass=10, nc=3, nc_hidden=64, sigma=0.1, grid_size=0.1, T=1.0)
    x = torch.randn(13, 3, 32, 32)
    out = classifier(x)
    print(out.size())
