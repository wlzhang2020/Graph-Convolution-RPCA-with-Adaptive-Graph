import scipy.io as scio
import os
from Data_Process import *
import torch

dataset_list = ['UMIST']

for dataset in dataset_list:
    path1 = os.path.abspath('.')
    path = '{}/data/{}.mat'.format(path1, dataset)
    save_path = '{}/data_occ/{}.mat'.format(path1, dataset)
    data = scio.loadmat(path)
    if 'Y' in data:
        labels = data['Y']
    else:
        labels = data['label']

    features = data['X']
    features = features / 1.0
    features = torch.Tensor(features)
    if features.shape[1] == labels.shape[0]:
        features = torch.transpose(features, 1, 0)  # change the data into n×d

    normalized = Normalized(features)
    features = normalized.Normal()
    # features = normalized.MinMax()
    features_occ = get_occ(features, r1=0.2, r2=0.2)

    if 'W' in data:
        adj = data['W']
    elif 'S' in data:
        adj = data['S']
    else:
        print('Building graph by CAN.')
        graph = Graph(torch.Tensor(features_occ))
        adj = graph.CAN(15)

    scio.savemat(save_path, {'X': features.numpy(), 'Y': labels, 'Xocc': features_occ.numpy(), 'adj': adj.numpy()})
    print('The {} data set has been noised successfully.'.format(dataset))



