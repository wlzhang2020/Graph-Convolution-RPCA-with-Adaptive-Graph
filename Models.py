import torch
import numpy as np
import scipy.io as scio
import utils
from metrics import cal_clustering_metric
from sklearn.cluster import KMeans
from utils import *


class AdaGRPCA(torch.nn.Module):
    def __init__(self, raw_X, X, labels, layers=None, lam1=0.1, lam2=0.1, lam3=1e-6, num_neighbors=3, sigma=0.1, p=1,
                 learning_rate=10**-3, max_iter=50, max_epoch=10, update=True, inc_neighbors=2, links=0, device=None):
        super(AdaGRPCA, self).__init__()
        if layers is None:
            layers = [1024, 256, 64]
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.raw_X = raw_X
        self.X = X
        self.n_clusters = np.unique(labels).shape[0]
        self.labels = np.array(labels).flatten()
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.sigma = sigma
        self.p = p
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.num_neighbors = num_neighbors + 1
        self.embedding_dim = layers[-1]
        self.mid_dim = layers[1]
        self.input_dim = layers[0]
        self.update = update
        self.inc_neighbors = inc_neighbors
        self.max_neighbors = self.cal_max_neighbors()
        self.links = links
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

        self.loss_list = []
        self.rce_list = []
        self.x_list = []
        self.embedding = None
        self._build_up()

    def cal_D(self, X):
        # print('X', X)
        eps = 1e-6
        # m = X.shape[1]
        # I = (eps * torch.eye(m)).to(self.device)
        T = torch.matmul(X.t(), X)
        e, v = torch.symeig(T, eigenvectors=True)
        e = torch.relu(e) + eps
        # print('e', e)
        e = e ** ((self.p - 2) / 2)
        # print('e_pow', e)
        # print('v', v)
        T_pow = v.matmul((torch.diag(e)).matmul(v.t()))
        D = 0.5 * self.p * T_pow
        D = D.detach()
        # print('D', D)
        return D

    def update_D(self, X):
        self.D = self.cal_D(X)

    def _build_up(self):
        self.W1 = get_weight_initial([self.input_dim, self.mid_dim])
        self.W2 = get_weight_initial([self.mid_dim, self.embedding_dim])
        self.W3 = get_weight_initial([self.embedding_dim, self.mid_dim])
        self.W4 = get_weight_initial([self.mid_dim, self.input_dim])

    def sigma_norm(self, X):
        size = torch.numel(X)
        tmp = torch.norm(X, dim=1)
        res = torch.sum(tmp*tmp*(1+self.sigma)/(tmp+self.sigma)) / size
        return res

    def cal_max_neighbors(self):
        if not self.update:
            return 0
        size = self.X.shape[0]
        num_clusters = np.unique(self.labels).shape[0]
        return 1.0 * size / num_clusters

    def forward(self, Laplacian):
        # sparse
        embedding = Laplacian.mm(self.X.matmul(self.W1))
        embedding = torch.nn.functional.relu(embedding)
        # sparse
        self.embedding = Laplacian.mm(embedding.matmul(self.W2))
        distances = utils.distance(self.embedding.t(), self.embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)

        # decoder 2
        # x_3 = Laplacian.matmul(self.embedding.matmul(self.W3))
        # x_3 = torch.nn.functional.relu(x_3)
        # recons_x = Laplacian.matmul(x_3.matmul(self.W4))

        # decoder FC
        x_3 = torch.nn.functional.relu(self.embedding.matmul(self.W3))
        x_4 = x_3.matmul(self.W4)
        recons_x = x_4

        # sparseProb = SparseProb(sparsity=self.num_neighbors)
        # recons_w = sparseProb(distances)
        return recons_w + 1e-10, recons_x + 1e-10
        # return 1 / (distances + 1)
        # return torch.sigmoid(self.embedding.matmul(torch.t(self.embedding)))

    def update_graph(self):
        weights, raw_weights = utils.cal_weights_via_CAN(self.embedding.t(), self.num_neighbors, self.links)  # first
        weights = weights.detach()
        raw_weights = raw_weights.detach()
        # threshold = 0.5
        # connections = (recons > threshold).type(torch.IntTensor).cuda()
        # weights = weights * connections
        Laplacian = utils.get_Laplacian_from_weights(weights)
        # Laplacian = utils.get_Laplacian_from_weights(utils.noise(weights))
        return weights, Laplacian, raw_weights

    def update_graph_entropy(self, recons):
        size = self.embedding.shape[0]
        distances = utils.distance(self.embedding.t(), self.embedding.t())
        distances = self.lam1 * distances - recons.log()
        distances = distances.detach()
        distances = torch.exp(-distances)
        sorted_distances, _ = distances.sort(dim=1, descending=True)
        flags = sorted_distances[:, self.num_neighbors].reshape([size, 1])
        distances[distances <= flags] = 0
        sums = distances.sum(dim=1).reshape([size, 1])
        raw_weights = distances / sums
        weights = (raw_weights + raw_weights.t()) / 2
        Laplacian = utils.get_Laplacian_from_weights(weights)
        return weights, Laplacian, raw_weights

    def build_loss(self, recons_w, recons_x, weights, raw_weights):
        size = self.X.shape[0]
        loss1 = raw_weights * torch.log(raw_weights / recons_w + 10**-10)
        loss1 = loss1.sum(dim=1)
        loss1 = loss1.mean()
        # loss += 10**-3 * (torch.mean(self.embedding.pow(2)))
        # loss += 10**-3 * (torch.mean(self.W1.pow(2)) + torch.mean(self.W2.pow(2)))
        # loss += 10**-3 * (torch.mean(self.W1.abs()) + torch.mean(self.W2.abs()))
        degree = weights.sum(dim=1)
        L = torch.diag(degree) - weights
        loss1 += self.lam1 * torch.trace(self.embedding.t().matmul(L).matmul(self.embedding)) / size
        # loss2 = self.loss_fn(self.X, recons_x) + self.lam2 * torch.norm(recons_x, 'nuc')

        # loss2 = self.sigma_norm(self.X-recons_x) + self.lam2 * torch.norm(recons_x, 'nuc')

        sp_norm = torch.trace(recons_x.matmul(torch.matmul(self.D, recons_x.t())))
        # sp_norm = torch.pow(torch.trace(recons_x.t().matmul(recons_x)), 0.5 * self.p)
        # loss2 = self.loss_fn(self.X, recons_x) + self.lam2 * sp_norm
        loss2 = self.sigma_norm(self.X - recons_x) + self.lam2 * sp_norm

        if self.lam3 > 1e15:
            loss1 = 0
        else:
            loss2 *= self.lam3
        # print('loss1: %5.4f loss2: %5.4f' % (loss1, loss2))
        return loss1 + loss2

    def run(self):
        weights, raw_weights = utils.cal_weights_via_CAN(self.X.t(), self.num_neighbors, self.links)
        Laplacian = utils.get_Laplacian_from_weights(weights)
        Laplacian = Laplacian.to_sparse()
        torch.cuda.empty_cache()
        # Laplacian = utils.get_Laplacian_from_weights(utils.noise(weights))
        print('Raw-CAN:', end=' ')
        res_metrics = self.evalutaion(weights, recons_x=None, k_means=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)
        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                # optimizer.zero_grad()
                recons_w, recons_x = self(Laplacian)
                if i % 1 == 0:
                    self.update_D(recons_x)
                loss = self.build_loss(recons_w, recons_x, weights, raw_weights)
                weights = weights.cpu()
                raw_weights = raw_weights.cpu()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)
                if i % 50 == 0:
                    self.loss_list.append(loss.cpu().detach().numpy())
                    self.rce_list.append(np.linalg.norm(self.raw_X-recons_x.cpu().detach().numpy()) ** 2)
                    self.x_list.append(epoch * self.max_iter + i)
                # print('epoch-%3d-i:%3d,' % (epoch, i), 'loss: %6.5f' % loss.item())
            # scio.savemat('results/embedding_{}.mat'.format(epoch), {'Embedding': self.embedding.cpu().detach().numpy()})
            if self.num_neighbors < self.max_neighbors:
                weights, Laplacian, raw_weights = self.update_graph()
                # weights, Laplacian, raw_weights = self.update_graph_entropy(recons)
                # self.clustering(weights, k_means=False)
                res_metrics = self.evalutaion(weights, recons_x, k_means=True, SC=True)
                self.num_neighbors += self.inc_neighbors
            else:
                if self.update:
                    self.num_neighbors = int(self.max_neighbors)
                    break
                recons = None
                weights = weights.cpu()
                raw_weights = raw_weights.cpu()
                torch.cuda.empty_cache()
                # w, _, __ = self.update_graph()
                # _, __ = (None, None)
                # torch.cuda.empty_cache()
                # w, _, __ = self.update_graph_entropy(recons)
                res_metrics = self.evalutaion(weights, recons_x, k_means=False)
                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)
                if self.update:
                    break

            # print('epoch:%3d,' % epoch, 'loss: %6.5f' % loss.item())
            # print('epoch:%3d,' % epoch, 'the rank of reconstructed X', torch.matrix_rank(recons_x))
        return res_metrics

    def evalutaion(self, weights, recons_x=None, k_means=True, SC=True):
        acc_km, nmi_km, pur_km, acc_sc, nmi_sc, pur_sc, recon_err = 0, 0, 0, 0, 0, 0, 0
        n_clusters = self.n_clusters
        if k_means:
            embedding = self.embedding.cpu().detach().numpy()
            km = KMeans(n_clusters=n_clusters).fit(embedding)
            prediction = km.predict(embedding)
            # print(np.max(embedding))
            acc, nmi, pur = cal_clustering_metric(self.labels, prediction)
            acc_km, nmi_km, pur_km = acc, nmi, pur
            print('k-means --- ACC: %5.4f, NMI: %5.4f PUR: %5.4f' % (acc, nmi, pur), end=' ')
        if SC:
            degree = torch.sum(weights, dim=1).pow(-0.5)
            L = (weights * degree).t() * degree
            L = L.cpu()
            _, vectors = L.symeig(True)
            indicator = vectors[:, -n_clusters:]
            indicator = indicator / (indicator.norm(dim=1)+10**-10).repeat(n_clusters, 1).t()
            indicator = indicator.cpu().numpy()
            km = KMeans(n_clusters=n_clusters).fit(indicator)
            prediction = km.predict(indicator)
            acc, nmi, pur = cal_clustering_metric(self.labels, prediction)
            acc_sc, nmi_sc, pur_sc = acc, nmi, pur
            print('SC --- ACC: %5.4f, NMI: %5.4f PUR: %5.4f' % (acc, nmi, pur), end=' ')
        if recons_x is not None:
            recon_err = np.linalg.norm(self.raw_X-recons_x.cpu().detach().numpy()) ** 2
            print('RCE: %5.4f ' % recon_err, end='')
        print('')
        return acc_km, nmi_km, pur_km, acc_sc, nmi_sc, pur_sc, recon_err


class SparseProb(torch.nn.Module):
    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, distances):
        """
        :param distances: n * n
        :return:
        """
        size = distances.shape[0]
        sorted_distances, _ = distances.sort(dim=1)
        top_k = sorted_distances[:, self.sparsity]
        top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

        sum_top_k = torch.sum(sorted_distances[:, 0:self.sparsity], dim=1)
        sum_top_k = torch.t(sum_top_k.repeat(size, 1))
        prob = torch.div(top_k - distances, self.sparsity * top_k - sum_top_k)
        return prob.relu()


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)
