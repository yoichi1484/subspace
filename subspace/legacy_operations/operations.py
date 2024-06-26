import numpy as np
from scipy.linalg import orth

def subspace_np(A):
    """ Compute orthonormal bases of the subspace
        Args:
            A: bases of the linear subspace (n_bases, dim)
        Return:
            Orthonormal bases 
        Example:
            >>> A = np.random.random_sample((10, 300)) 
            >>> subspace_np(A)
    """
    return orth(A.T).T

    
def intersection_np(SA, SB, threshold=1e-2):
    """ Compute bases of the intersection
        Args:
            SA, SB: bases of the linear subspace (n_bases, dim)
        Return:
            Bases of intersection
        Example:
            >>> A = np.random.random_sample((10, 300)) 
            >>> B = np.random.random_sample((20, 300)) 
            >>> intersection_np(A, B)
    """
    assert threshold > 1e-6
    
    if SA.shape[0] > SB.shape[0]:
        return intersection_np(SB, SA, threshold)
    
    # orthonormalize
    SA = subspace_np(SA)
    SB = subspace_np(SB)

    # compute canonical angles
    u, s, v = np.linalg.svd(SA @ SB.T)

    # extract the basis that the canonical angle is zero
    u = u[:, np.abs(s - 1.0) < threshold]
    return (SA.T @ u).T


def sum_space_np(SA, SB):    
    """ Compute bases of the sum space
        Args:
            SA, SB: bases of the linear subspace (n_bases, dim)
        Return:
            Bases of sum space
        Example:
            >>> A = np.random.random_sample((10, 300)) 
            >>> B = np.random.random_sample((20, 300)) 
            >>> sum_space_np(A, B)
    """
    M = np.concatenate([SA, SB], axis=0)
    return subspace_np(M)


def orthogonal_complement_np(SA, threshold=1e-2):
    """ Compute bases of the orthogonal complement
        Args:
            SA: bases of the linear subspace (n_bases, dim)
        Return:
            Bases of the orthogonal complement
        Example:
            >>> A = np.random.random_sample((10, 300)) 
            >>> orthogonal_complement_np(A)
    """
    assert threshold > 1e-6
    u, s, v = np.linalg.svd(SA.T)
    # compute rank
    rank = (s > threshold).sum()
    return u[:, rank:].T


def soft_membership_np(A, v):
    """ Compute membership degree of the vector v for the subspace A
        Args:
            A: bases of the linear subspace (n_bases, dim)
            v: vector (dim,)
        Return:
            soft membership degree
        Example:
            >>> A = np.array([[1,0,0], [0,1,0]])
            >>> v = np.array([1,0,0])
            >>> soft_membership_np(A, v)
            1.0
            >>> A = np.array([[1,0,0], [0,1,0]])
            >>> v = np.array([0,0,1])
            >>> soft_membership_np(A, v)
            0.0
    """
    v = v.reshape(1, len(v))
    v = subspace_np(v)
    A = subspace_np(A)
    
    # The cosine of the angles between a subspace and a vector are singular values
    u, s, v = np.linalg.svd(A @ v.T) 
    s[s > 1] = 1
    
    # Return the maximum cosine of the canonical angles, i.e., the soft membership.
    return np.max(s)
