from Models import AdaGRPCA
from DataLoader import *
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

dataset = 'UMIST'
data_loader = LoadDataOcc(dataset)
X, labels, X_occ = data_loader.load()

input_dim = X.shape[1]
layers = [input_dim, 256, 10]
lam1 = 1
lam2 = 1e-6
lam3 = 1e2
k_neighbors = 5
lr = 1e-3
t0 = 200
T = 5
inc_neighbors = 2
p = 2/3
sigma = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_occ = torch.Tensor(X_occ).to(device)
model = AdaGRPCA(X, X_occ, labels, layers=layers, num_neighbors=k_neighbors, lam1=lam1, lam2=lam2,
                 lam3=lam3, sigma=sigma, p=p, max_iter=t0, max_epoch=T, update=True, learning_rate=lr,
                 inc_neighbors=inc_neighbors, device=device)
model.run()

