import scipy.io as scio
import torch
import os
import numpy as np
import scipy as sp
import copy
from random import sample, uniform

# load the data after noising.
class LoadData():
    def __init__(self, dataset):
        self.dataset = dataset

    def CPU(self):
        path1 = os.path.abspath('.')
        path = '{}/data_occ/{}.mat'.format(path1, self.dataset)
        data = scio.loadmat(path)
        features = torch.Tensor(data['X'])
        labels = torch.Tensor(data['Y'])
        features_occ = torch.Tensor(data['Xocc'])
        adj = torch.Tensor(data['adj'])
        return features, labels, features_occ, adj


# load the datasets
class Load_Data():
    def __init__(self, name):
        self.name = name

    def CPU(self):
        path1 = os.path.abspath('.')
        path = '{}/data/{}.mat'.format(path1, self.name)
        data = scio.loadmat(path)
        if 'Y' in data:
            labels = torch.Tensor(data['Y'])
        else:
            labels = torch.Tensor(data['label'])
        #labels = torch.reshape(labels, (labels.shape[0],))
        features = data['X']
        features = features / 1.0
        features = torch.Tensor(features)
        if features.shape[1] == labels.shape[0]:
           features = torch.transpose(features, 1, 0)  # change the data into n×d
        if 'W' in data:
            adj = data['W']
        elif 'S' in data:
            adj = data['S']
        else:
            print('Building graph by CAN.')
            graph = Graph(features)
            adj = graph.CAN(15)
            #adj = sp.coo_matrix(adj)
        return torch.Tensor(features), torch.Tensor(labels), torch.Tensor(adj)

    def GPU(self):
        #
        path1 = os.path.abspath('.')
        path = '{}/data/{}.mat'.format(path1, self.name)
        #
        #path = '/Users/wutong/Downloads/Code_New/Dataset/{}.mat'.format(self.name)
        data = scio.loadmat(path)
        labels = torch.Tensor(data['Y'])
        labels = torch.reshape(labels, (labels.shape[0],))
        features = data['X']
        features = features / 1.0
        features = torch.Tensor(features)
        if features.shape[1] == labels.shape[0]:
            features = torch.transpose(features, 1, 0)  # 把数据变成n×d
        if 'W' in data:
            adj = data['W']
        elif 'S' in data:
            adj = data['S']
        else:
            graph = Graph(features)
            adj = graph.CAN(15)
            #adj = sp.sparse.coo_matrix(adj)
        return torch.Tensor(features), torch.Tensor(labels), torch.Tensor(adj)

class Normalized():
    def __init__(self, X):
        self.X = X

    def Normal(self):
        # X: n×d，normalize each dimension of X
        return (self.X - torch.mean(self.X, 0, keepdim=True)) / (torch.std(self.X, 0, keepdim=True) + 1e-4)

    def Length(self):
        # X:n*d, Make each row of X equal to each other.
        return self.X / (torch.sum(self.X, 1, keepdim=True) + 1e-6)

    def MinMax(self):
        # Normalize matrix by Min and Max
        # X: n*d
        min, _ = torch.min(self.X, 0, keepdim=True)
        max, _ = torch.max(self.X, 0, keepdim=True)
        return (self.X - min) / (max - min)

    def Normal_Row(self):
        # X: n×d，normalize each row of X
        return (self.X - torch.mean(self.X, 1, keepdim=True)) / (torch.std(self.X, 1, keepdim=True) + 1e-4)


class Graph():
    def __init__(self, X):
        self.X = X

    def Middle(self):
        Inner_product = self.X.mm(self.X.T)
        # row_normalized = BZH(Inner_product)
        Graph_middle = torch.sigmoid(Inner_product)
        return Graph_middle

    # Construct the adjacency matrix by CAN
    def CAN(self, k):
        # Input: n * d. I change it to d*n in this subfunction
        self.X = self.X.T
        n = self.X.shape[1]
        D = L2_distance_2(self.X, self.X)
        _, idx = torch.sort(D)
        S = torch.zeros(n, n)
        for i in range(n):
            id = torch.LongTensor(idx[i][1:k + 1 + 1])
            di = D[i][id]
            S[i][id] = (torch.Tensor(di[k].repeat(di.shape[0])) - di) / (k * di[k] - torch.sum(di[0:k]) + 1e-4)
        S = (S + S.T) / 2
        return S

    # Reconstruct adjacency matrix by Rui Zhang
    def Reconstruct(self, beta):
        # X : n*d
        n, d = self.X.size()
        # X = BZH_Length(X)
        # A = BZH(X.mm(X.T))
        A = (self.X).mm(self.X.T)
        A[A < 1e-4] = 0
        F = torch.sigmoid(A)
        S = torch.zeros(n, n)
        E = L2_distance_2(self.X.T, self.X.T)
        A_alpha = (F - beta * E)
        for i2 in range(n):
            tran = EProjSimplex_new(A_alpha[:, i2:i2 + 1], 1)
            S[:, i2:i2 + 1] = tran
        S = (S + S.T) / 2
        return S

    # Construct the adjacency matrix by Gaussian.
    # Input X：n×d
    # Failed, don't use
    def Gaussian(self, k=10):
        D = L2_distance_2(self.X.T, self.X.T)
        sort_D, index = torch.sort(D)
        index_k = index[:, k - 1:k]
        D_index = torch.gather(D, dim=1, index=index_k)
        S = D
        S[S > D_index] = 0
        S = S / (torch.sum(S, 1, keepdim=True) + 1e-6)
        S = S - torch.diag(torch.diag(S))
        return (S + S.T) / 2

class Adjacency():
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def Modified(self):
        adj = self.adjacency + torch.eye(self.adjacency.shape[0])
        degrees = torch.Tensor(adj.sum(1))
        degrees_matrix_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return torch.mm(degrees_matrix_inv_sqrt, adj).mm(degrees_matrix_inv_sqrt)

class Laplacian():
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def Original(self):
        degrees = torch.diag(torch.Tensor(self.adjacency.sum(1)).flatten())
        return degrees - self.adjacency

    # Normalized Laplacian matrix
    def Modified(self):
        degrees = (torch.Tensor(self.adjacency.sum(1)).flatten())
        L = self.adjacency
        D_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return D_sqrt.mm(L).mm(D_sqrt)

    def Normalized(self):
        degrees = (torch.Tensor(self.adjacency.sum(1)).flatten())
        D = torch.diag(degrees)
        L = D  - self.adjacency
        D_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return D_sqrt.mm(L).mm(D_sqrt)

# The construction of adjacency matrix， X：n×d
def SpectralClustering(L, f):
    #similarity = Graph_Gaussian(X)
    #similarity = torch.Tensor(similarity)
    #similarity = torch.abs((similarity + similarity.T)/2)
    #L = get_modified_Laplacian(similarity)
    eigenvalues, eigenvectors = torch.eig(L, eigenvectors=True)
    eigenvalues = eigenvalues[:, 0]
    sorted, indices = torch.sort(eigenvalues)
    indices = indices[0:f]
    W = eigenvectors[:, indices]
    W_std = Normalized(W)
    W_std = W_std.Normal_Row()
    return W_std

# Construct the adjacency matrix by inner and add the sigmoid
# The initialization of the weighted matrix
def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2*bound*torch.rand(d1, d2)
    return torch.Tensor(nor_W)

# L_2 norm squared between matrix samples
# each column is a sample
def L2_distance_2(A, B):
    # A = A.detach().numpy()
    # B = B.detach().numpy()
    AA = torch.sum(A*A, dim=0, keepdims=True)
    BB = torch.sum(B*B, dim=0, keepdims=True)
    AB = (A.T).mm(B)
    #D = np.tile(torch.transpose(AA, 1, 0), (1, BB.shape[1])) + np.tile(BB, (AA.shape[1], 1)) - 2*AB
    D = ((AA.T).repeat(1, BB.shape[1])) + (BB.repeat(AA.shape[1], 1)) - 2 * AB
    #D = torch.real(D)
    #D = D.clip(min=0)
    D = torch.abs(D)
    return D

def EProjSimplex_new(v, k=1):
    ft = 1
    n = v.shape[0]
    v0 = v-torch.mean(v) + k/n
    vmin = torch.min(v0)
    v1_0 = torch.zeros(n, 1)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-4:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = sum(posidx.float())
            g = -npos
            f = sum(v1[posidx])-k
            lambda_m = lambda_m-f/(g+1e-6)
            ft = ft+1
            v1_0 = v1_0.type(v1.type())
            if ft > 100:
                x = torch.max(v1, v1_0)
                break
        x = torch.max(v1, v1_0)
    else:
        x = v0
    return x


def PCA(dataMat, topNfeat=999999):
    meanVals = np.mean(dataMat, axis=0)
    DataAdjust = dataMat - meanVals
    covMat = np.cov(DataAdjust, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    #print eigVals
    eigVals, eigVects = eigVals.real, eigVects.real
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = np.dot(DataAdjust, redEigVects)
    reconMat = np.dot(lowDDataMat, redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def get_occ(X, r1=0.2, r2=0.2):
    [n, dim] = X.shape
    X_occ = copy.deepcopy(X)
    max_ele = torch.max(torch.max(X))
    min_ele = torch.min(torch.min(X))

    mask_sample = sample(range(n), int(r1*n))
    for i in mask_sample:
        mask_feature = sample(range(dim), int(r2*dim))
        for j in mask_feature:
            X_occ[i][j] = uniform(min_ele, max_ele)

    # test
    # for i in range(n):
    #     sum = 0
    #     for j in range(dim):
    #         if X[i][j] != X_occ[i][j]:
    #             sum += 1
    #     print(sum / dim)

    return X_occ




