import numpy as np
import scipy


def grassmann_distance(U, V):
    """ Compute geodesic distance for grassmann manifold 
    
        Args:
            U, V: A matrix of bases of a linear subspace
        Return:
            grassmann distance
        See Also:
            scipy.linalg.subspace_angles
        Example:
            >>> U = np.array([[1,0,0], [1,1,1]])
            >>> V = np.array([[0,1,0], [1,1,1]])
            >>> grassmann_distance(U, V)
    """    
    # compute the canonical angles
    s = scipy.linalg.subspace_angles(U.T, V.T)
    # grassmann distance
    return sum(s * s)


def grassmann_similarity(x, y):
    return -grassmann_distance(x, y)