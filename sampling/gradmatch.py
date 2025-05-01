import numpy as np
from scipy.linalg import solve
from itertools import batched

from .utils import timeit


def OrthogonalMP(A, b, tol=1e-4, nnz=None, positive=False):

    AT = A.T
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = np.zeros(n)
    resid = np.copy(b)
    normb = np.linalg.norm(b)
    indices = []

    for i in range(nnz):
        if (np.linalg.norm(resid) / normb) < tol:
            break
        projections = AT.dot(resid)
        if positive:
            index = np.argmax(projections)
        else:
            index = np.argmax(abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / A_i.T.dot(A_i)
        else:
            A_i = np.vstack([A_i, A[:, index]])
            x_i = solve(A_i.dot(A_i.T), A_i.dot(b), assume_a="sym")
            if positive:
                while min(x_i) < 0.0:
                    argmin = np.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1 :]
                    A_i = np.vstack([A_i[:argmin], A_i[argmin + 1 :]])
                    x_i = solve(A_i.dot(A_i.T), A_i.dot(b), assume_a="sym")
        resid = b - A_i.T.dot(x_i)

    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x


@timeit
def gradmatch(
    dataset,
    K,
    tol=1e-4,
    batch_size=1024,
):
    """
    GradMatch algorithm for selecting a subset of data points based on their gradients.

    Parameters:
        X (numpy.ndarray): The input data matrix.
        y (numpy.ndarray): The target labels.
        n_samples (int): The number of samples in the dataset.
        n_features (int): The number of features in the dataset.
        n_classes (int): The number of classes in the dataset.
        alpha (float): Regularization parameter for the gradient matching.
        max_iter (int): Maximum number of iterations for the optimization.
        tol (float): Tolerance for convergence.
        batch_size (int): Size of the batches for processing.

    Returns:
        numpy.ndarray: The selected subset of data points based on their gradients.
    """
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # y = kmeans.fit_predict(dataset).reshape(-1, 1)
    sset = []
    gammas = []
    idx = np.arange(len(dataset))
    batches = zip(batched(idx, batch_size), batched(dataset, batch_size))
    for tmp_idx, batch in batches:
        batch = np.array(batch)
        tmp_idx = np.array(tmp_idx)
        grad = np.gradient(batch, axis=1, edge_order=2)
        grad_sum = np.sum(grad, axis=0)
        # Compute the gradient matching for each batch
        r = OrthogonalMP(grad.T, grad_sum, tol=tol, nnz=K // batch_size)
        selected_indices = tmp_idx[np.abs(r) > 0.5]
        sset.extend(selected_indices)

    sset = np.array(sset)

    diff = K - len(sset)
    if diff > 0:
        remainList = set(np.arange(len(dataset))).difference(set(sset))
        new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
        sset = np.concatenate((sset, new_idxs))

    return sset
