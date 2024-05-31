import torch

def subspace(A: torch.Tensor) -> torch.Tensor:
    """
    Compute orthonormal bases of the subspace
        Args:
            A: bases of the linear subspace (n_bases, dim)
        Return:
            Orthonormal bases
        Example:
            >>> A = torch.rand(10, 300)
            >>> subspace(A)
    """
    return torch.linalg.qr(A.t()).Q.t()


def intersection(SA: torch.Tensor, SB: torch.Tensor, threshold: float = 1e-2) -> torch.Tensor:
    """
    Compute bases of the intersection
        Args:
            SA, SB: bases of the linear subspace (n_bases, dim)
        Return:
            Bases of intersection
        Example:
            >>> A = torch.rand(10, 300)
            >>> B = torch.rand(20, 300)
            >>> intersection(A, B)
    """
    assert threshold > 1e-6

    if SA.shape[0] > SB.shape[0]:
        return intersection(SB, SA, threshold)

    # orthonormalize
    SA = subspace(SA)
    SB = subspace(SB)

    # compute canonical angles
    u, s, v = torch.linalg.svd(SA @ SB.t())

    # extract the basis that the canonical angle is zero
    u = u[:, (s - 1.0).abs() < threshold]
    return (SA.t() @ u).t()


def sum_space(SA: torch.Tensor, SB: torch.Tensor) -> torch.Tensor:
    """
    Compute bases of the sum space
        Args:
            SA, SB: bases of the linear subspace (n_bases, dim)
        Return:
            Bases of sum space
        Example:
            >>> A = torch.rand(10, 300)
            >>> B = torch.rand(20, 300)
            >>> sum_space(A, B)
    """
    M = torch.cat([SA, SB], dim=0)
    return subspace(M)


def orthogonal_complement(SA: torch.Tensor, threshold: float = 1e-2) -> torch.Tensor:
    """
    Compute bases of the orthogonal complement
        Args:
            SA: bases of the linear subspace (n_bases, dim)
        Return:
            Bases of the orthogonal complement
        Example:
            >>> A = torch.rand(10, 300)
            >>> orthogonal_complement(A)
    """
    assert threshold > 1e-6
    u, s, v = torch.linalg.svd(SA.t())
    # compute rank
    rank = (s > threshold).sum()
    return u[:, rank:].T


def soft_membership(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute membership degree of the vector v for the subspace A
        Args:
            A: bases of the linear subspace (n_bases, dim)
            v: vector (dim,)
        Return:
            soft membership degree
        Example:
            >>> A = torch.tensor([[1,0,0], [0,1,0]])
            >>> v = torch.tensor([1,0,0])
            >>> soft_membership(A, v)
            1.0
            >>> A = torch.tensor([[1,0,0], [0,1,0]])
            >>> v = torch.tensor([0,0,1])
            >>> soft_membership(A, v)
            0.0
    """
    v = v.reshape(1, len(v))
    v = subspace(v)
    A = subspace(A)

    # The cosine of the angles between a subspace and a vector are singular values
    u, s, v = torch.linalg.svd(A @ v.t())
    s[s > 1] = 1

    # Return the maximum cosine of the canonical angles, i.e., the soft membership.
    return torch.max(s)

