import numpy as np

def pagerank(M, n_iter: int = 100, d: float = 0.85):
    """Google's PageRank algorithm. Google's Founders -- Sergey Brin and Larry Page -- proposed this algorithm
    to calculate the importance of website pages based on search engine results. Building on that,
    this function would return the ranking of nodes (webpages) from a given adjacency matrix M using a power method.
    
    
    See: https://en.wikipedia.org/wiki/PageRank
    
    Parameters:
    -----------
    M : np.array
        The adjacency matrix of the network of webpages/nodes. A transition matrix - sum of all path to 
        node i from all other nodes would be 1.
    n_iter : int, default = 100
        The number of iteration for steady-state values of scores.
    d : float, default = 0.85
        Damping factor -- somewhat a "ghost" surfer to give randomly
        give importance to other nodes (for inclusion).
        
    Returns:
    --------
    pgrank = np.array
        An array of ranks for each node in the network. Note that
        pgrank should be normalized (sum of rank values = 1)
    """
    # Get the number of nodes in the network
    n = M.shape[1]
    
    # Initialize the ranking vector -- equal rank for all nodes
    pgrank = np.ones(n) / n # Initial uniform probablity distribution
    
    # Calculate differrent probabilities of lading on other webpages based on adjacency
    M_hat = (d * M + (1 - d)/n) 
    
    # The page rank would be the principal eigenvector of M_hat @ pg_rank
    # Apply power mthod to copmute for the principal eigenvector after n_iter iterations
    for _ in range(n_iter):
        # Power method
        pgrank = M_hat @ pgrank
        
    return pgrank


'''Sample test case
M = np.array([[0, 0, 0, 0, 1],
              [0.5, 0, 0, 0, 0],
              [0.5, 0, 0, 0, 0],
              [0, 1, 0.5, 0, 0],
              [0, 0, 0.5, 1, 0]])
print(pagerank(M))

# Known pagerank answer: [0.2542, 0.1380, 0.1380, 0.2060, 0.2638]
pagerank(M)
'''