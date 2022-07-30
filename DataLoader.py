import os
import torch
import scipy.io as scio
from utils import cal_weights_via_CAN


# load the data after noising.
class LoadDataOcc:
    def __init__(self, dataset):
        self.dataset = dataset

    def load(self):
        print('load occluded dataset: {}'.format(self.dataset))
        path1 = os.path.abspath('.')
        path = '{}/data_occ/{}.mat'.format(path1, self.dataset)
        data = scio.loadmat(path)
        features = torch.Tensor(data['X'])
        labels = torch.Tensor(data['Y'])
        features_occ = torch.Tensor(data['Xocc'])
        return features, labels, features_occ


# load the raw datasets
class LoadRawData:
    def __init__(self, dataset):
        self.dataset = dataset

    def load(self):
        print('load raw dataset: {}'.format(self.dataset))
        path1 = os.path.abspath('.')
        path = '{}/data/{}.mat'.format(path1, self.dataset)
        data = scio.loadmat(path)
        if 'Y' in data:
            labels = torch.Tensor(data['Y'])
        else:
            labels = torch.Tensor(data['label'])
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
            adj, raw_adj = cal_weights_via_CAN(features.t(), 15)
        return features, labels, adj
        # return torch.Tensor(features), torch.Tensor(labels), torch.Tensor(adj)