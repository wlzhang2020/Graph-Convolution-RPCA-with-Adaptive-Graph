import torch


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    # result = result.max(torch.zeros(result.shape).cuda())
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    # result = torch.max(result, result.t())
    return result


def cal_weights_via_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links is not 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()
    return weights, raw_weights


def get_Laplacian_from_weights(weights):
    # W = torch.eye(weights.shape[0]).cuda() + weights
    # degree = torch.sum(W, dim=1).pow(-0.5)
    # return (W * degree).t()*degree
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t()*degree


def noise(weights, ratio=0.1):
    sampling = torch.rand(weights.shape).cuda() + torch.eye(weights.shape[0]).cuda()
    sampling = (sampling > ratio).type(torch.IntTensor).cuda()
    return weights * sampling


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
            id = torch.LongTensor(idx.cpu()[i][1:k + 1 + 1])
            di = D[i][id].cpu()
            S[i][id] = (torch.Tensor(di[k].repeat(di.shape[0])) - di) / (k * di[k] - torch.sum(di[0:k]) + 1e-10)
        raw_S = S.cuda()
        S = ((S + S.T) / 2).cuda()
        return S, raw_S


if __name__ == '__main__':
    tX = torch.rand(3, 8)
    print(cal_weights_via_CAN(tX, 3))
