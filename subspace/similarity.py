import torch

def get_weights(A, B, weight):
    # get weights
    if weight == "L2":
        weights_A = torch.linalg.norm(A, dim=2) 
        weights_B = torch.linalg.norm(B, dim=2) 
    elif weight == "L1":
        weights_A = torch.linalg.norm(A, dim=2, ord=1) 
        weights_B = torch.linalg.norm(B, dim=2, ord=1)
    elif weight == "no":
        weights_A = torch.ones(A.size(0), A.size(1)).to(A.device)
        weights_B = torch.ones(B.size(0), B.size(1)).to(B.device)
    else:
        raise NotImplementedError
    return weights_A, weights_B


def pairwise_cosine_matrix(matrix1, matrix2):
    # matrix1 の shape は (batch_size, num_features1, emb_dim)
    # matrix2 の shape は (batch_size, num_features2, emb_dim)
    dot = torch.matmul(matrix1, matrix2.transpose(1, 2))
    matrix1_norm = torch.norm(matrix1, dim=-1, keepdim=True)
    matrix2_norm = torch.norm(matrix2, dim=-1, keepdim=True)
    norm = torch.matmul(matrix1_norm, matrix2_norm.transpose(1, 2))
    return dot / norm


def subspace(A):
    """ Return the matrix of the subspace
        Arg:
            A: Bases of a linear subspace (batchsize, num_bases, emb_dim)
        Return:
            S: Orthonormalized bases of a linear subspace (batchsize, num_bases, emb_dim)
        Example:
            >>> A = torch.randn(5, 4, 300)
            >>> subspace(A)
    """ 
    # orthonormalize
    S, _ = torch.linalg.qr(torch.transpose(A, 1, 2))
    return torch.transpose(S, 1, 2)


@torch.jit.script
def soft_membership(S, v):
    """ Compute soft membership degree between a subspace and a vector
        Args:
            S: Orthonormalized bases of a linear subspace (batchsize, num_bases, emb_dim)
            v: vector (batchsize, emb_dim)
        Return:
            soft_membership degree (batchsize,)
        Example:
            >>> S = torch.randn(5, 4, 300)
            >>> v = torch.randn(5, 300)
            >>> soft_membership(S, v)
    """    
    # normalize
    v = torch.nn.functional.normalize(v)
    v = v.view(v.size(0), v.size(1), 1)

    # compute SVD for cos(theta)
    m = torch.matmul(S, v)
    s = torch.linalg.svdvals(m.float()) # s is the sequence of cos(theta_i)
    return torch.mean(s, 1)
    
    
def subspace_johnson(A, B, weight="L2"):  
    """ Compute similarity between two vector sets (sentences)
        Args:
            A: Matrix of word embeddings for the first sentence
               (batchsize, num_bases, dim)
            B: Matrix of word embeddings for the second sentence
               (batchsize, num_bases, dim)
        Return:
            similarity between A and B (batchsize,)
        Example:
            >>> A = torch.randn(5, 3, 300)
            >>> B = torch.randn(5, 4, 300)
            >>> subspace_johnson(A, B)
    """        
    def numerator(U, V, weights):
        """
            U should be a matrix of word embeddings
            V should be a matrix of orthonormalized bases
        """
        softm = torch.stack([soft_membership(V, vec) 
                             for vec in torch.transpose(U, 0, 1)]) 
        softm = torch.transpose(softm, 0, 1) 
        return torch.sum(softm * weights, 1)
        
    # get weights
    weights_A, weights_B = get_weights(A, B, weight)
        
    # compute similarity
    x = numerator(A, subspace(B), weights_A) / torch.sum(weights_A, 1)
    y = numerator(B, subspace(A), weights_B) / torch.sum(weights_B, 1)
    return x + y



def subspace_bert_score(A, B, weight="L2"):  
    """ Compute similarity between two vector sets (sentences)
        Args:
            A: Matrix of word embeddings for the first sentence
               (batchsize, num_bases, dim)
            B: Matrix of word embeddings for the second sentence
               (batchsize, num_bases, dim)
        Return:
            similarity between A and B (batchsize,)
        Example:
            >>> A = torch.randn(5, 3, 300)
            >>> B = torch.randn(5, 4, 300)
            >>> subspace_bert_score(A, B)
    """        
    def numerator(U, V, weights):
        """
            U should be a matrix of word embeddings
            V should be a matrix of orthonormalized bases
        """
        softm = torch.stack([soft_membership(V, vec) 
                             for vec in torch.transpose(U, 0, 1)]) 
        softm = torch.transpose(softm, 0, 1) 
        return torch.sum(softm * weights, 1)
        
    # get weights
    weights_A, weights_B = get_weights(A, B, weight)
        
    # Cmpute P, R, F
    R = numerator(A, subspace(B), weights_A) / torch.sum(weights_A, 1) # R は SubspaceJohnson の 左項
    P = numerator(B, subspace(A), weights_B) / torch.sum(weights_B, 1) # P は SubspaceJohnson の 右項
    F = (2 * P * R) / (P + R)
    return P, R, F


def vanilla_bert_score(A, B, weight="L2"):  
    """ Compute similarity between two vector sets (sentences)
        Args:
            A: Matrix of word embeddings for the first sentence
               (batchsize, num_bases, dim)
            B: Matrix of word embeddings for the second sentence
               (batchsize, num_bases, dim)
        Return:
            similarity between A and B (batchsize,)
        Example:
            >>> A = torch.randn(5, 3, 300)
            >>> B = torch.randn(5, 4, 300)
            >>> vanilla_bert_score(A, B)
    """        
    def numerator(pairwise_cos, dim, weights):
        max_cos, _ = pairwise_cos.max(dim=dim)
        return torch.sum(max_cos * weights, 1) # (max_cos * weights).sum(dim=1)
        
    # get weights
    weights_A, weights_B = get_weights(A, B, weight)
        
    # Pairwise cosine
    pairwise_cos = pairwise_cosine_matrix(A, B)
        
    # Cmpute P, R, F
    R = numerator(pairwise_cos, 2, weights_A) / torch.sum(weights_A, 1) # R は SubspaceJohnson の 左項
    P = numerator(pairwise_cos, 1, weights_B) / torch.sum(weights_B, 1) # P は SubspaceJohnson の 右項
    F = (2 * P * R) / (P + R)
    return P, R, F