import numpy as np
import scipy.sparse as sp
import time
import pickle
import torch


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def normalize_adj_sparse(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj_sparse(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj_sparse(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def _numpy_to_sparse(adj, if_entity=False):
    """
    将6个邻接矩阵，转为coo_sparse表示
    @:param adj : dev_dev_dep_graph_adj.npy  or dev_dev_entity_graph_adj.npy
    """
    batch_size = adj.shape[0]
    adj_coo_matrixs = []
    for i in range(batch_size):
        if if_entity:
            adj_coo_matrixs.append((sp.coo_matrix(adj[i][0]), sp.coo_matrix(adj[i][1]), sp.coo_matrix(adj[i][2])))  # 对于entity_graph包含多种边类型
        else:
            adj_coo_matrix = sp.coo_matrix(adj[i])
            adj_coo_matrixs.append(adj_coo_matrix)
    return adj_coo_matrixs


def handle_adj():
    # dev_dev_dep_graph_adj = np.load('../prepro_data/dev_dev_dep_graph_adj.npy')
    # dev_train_dep_graph_adj = np.load('../prepro_data/dev_train_dep_graph_adj.npy')
    # dev_test_dep_graph_adj = np.load('../prepro_data/dev_test_dep_graph_adj.npy')
    dev_dev_entity_graph_adj = np.load('../prepro_data/dev_dev_entity_graph_adj.npy')
    dev_train_entity_graph_adj = np.load('../prepro_data/dev_train_entity_graph_adj.npy')
    dev_test_entity_graph_adj = np.load('../prepro_data/dev_test_entity_graph_adj.npy')
    # dev_dev_dep_graph_adj = _numpy_to_sparse(dev_dev_dep_graph_adj)
    # dev_train_dep_graph_adj = _numpy_to_sparse(dev_train_dep_graph_adj)
    # dev_test_dep_graph_adj = _numpy_to_sparse(dev_test_dep_graph_adj)
    dev_dev_entity_graph_adj = _numpy_to_sparse(dev_dev_entity_graph_adj, if_entity=True)
    dev_train_entity_graph_adj = _numpy_to_sparse(dev_train_entity_graph_adj, if_entity=True)
    dev_test_entity_graph_adj = _numpy_to_sparse(dev_test_entity_graph_adj, if_entity=True)
    # pickle.dump(dev_dev_dep_graph_adj, open('../prepro_data/dev_dev_dep_graph_adj.pkl', "wb"))
    # pickle.dump(dev_train_dep_graph_adj, open('../prepro_data/dev_train_dep_graph_adj.pkl', "wb"))
    # pickle.dump(dev_test_dep_graph_adj, open('../prepro_data/dev_test_dep_graph_adj.pkl', "wb"))
    pickle.dump(dev_dev_entity_graph_adj, open('../prepro_data/dev_dev_entity_graph_adj.pkl', "wb"))
    pickle.dump(dev_train_entity_graph_adj, open('../prepro_data/dev_train_entity_graph_adj.pkl', "wb"))
    pickle.dump(dev_test_entity_graph_adj, open('../prepro_data/dev_test_entity_graph_adj.pkl', "wb"))

if __name__ == "__main__":
    handle_adj()
