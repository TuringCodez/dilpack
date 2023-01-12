import numpy as np

def mahalanobis_dist(M, A):
    """Computes the Mahalonobis Distance between a matrix (n-dim) and another 
    distribution matrix.
    
    Parameters:
    -----------
    M : np.array
        The relevant data points (array/matrix)
    A : np.array
        Matrix containing the different distribution where the
        distance from M would be calculated.
    """
    # Compute the covariance matrix. As discussed in the MP prep forum
    # A good generalization of the calculation of the inverse of the matrix
    # is through the use of the pseudo-inverse (Moore-Penrose) if the inverse of the matrix
    # is non-existent.
    inv_mat = np.linalg.pinv(np.cov(A.T))
    
    # Subtract the mean of A from M
    shifted_mat = M - np.mean(A)
    
    # Mahalonobis distance
    m_dist = np.array(shifted_mat @ inv_mat) @ np.array(shifted_mat.T)
    
    if type(m_dist) == np.ndarray:
        return np.sqrt(np.diag(m_dist))
    else:
        return np.sqrt(m_dist)
    
def pred_class_mahalanobis(train_data, train_label, test_data):
    """Creates a classification model by utilizing Mahalanobis distance to the training data.
    Predicts the classes for the test_data using the trained model.
    
    Parameters:
    -----------
    train_data : np.array
        Training data for the classification exercise
    test_data : np.array
        Vector containing the labels for the train data
    test_data : np.array
        The data which we want to classify
        
    Returns:
    --------
    pred_label : np.array
        The predicted cluster labels for the test_data
    """
    # Store the labels and separate the data by clusters
    cl_labels = np.unique(train_label)
    cl_grp = []
    for cl_ in cl_labels:
        # create mask
        mask_temp = (train_label == cl_)
        
        # Filter data
        cl_grp.append(train_data[mask_temp,:])
        
    # Sub function for calculating prob
    def sub_func_prob_calc(M):      
        # Input the matrix to be converted into a probability matrix
        M_out = 1 - (M / M.sum(axis=1)[:, np.newaxis])
        
        # Normalize
        M_out = M_out / M_out.sum(axis=1)[:, np.newaxis]
        
        return M_out
    
    # Calculate the distance to each cluster from the training
    for i, cl_ in enumerate(cl_labels):
        cl_grp_temp = cl_grp[i]
        
        # Calculate distance to the cluster
        dist_ = mahalanobis_dist(test_data, cl_grp_temp)
        
        # Append
        if i == 0:
            dist_array = dist_
        else:
            dist_array = np.column_stack((dist_array, dist_))
    
    # Convert distances to probabilities
    prob_mat = sub_func_prob_calc(dist_array)
    
    # Get prediction
    pred_label = np.array([cl_labels[np.argmax(row)] for row in prob_mat])
    
    return pred_label   

'''Sample Test Cases
# Distance calculation example
# From forum discussion - sample case where inverse is non existeent - edge case
A = np.array([[1., 100., 10.],
[2., 300., 15.],
[4., 200., 20.],
[2., 600., 10.],
[5., 100., 30.]])

X = np.array([6., 500., 40.]) 

mahalanobis_dist(X,A)

# Classifier example

# data path
data_path = '../../../example/classifier/mahalonobis'

# Load
train = np.load(data_path + '/train.npy')
train_labels = np.load(data_path + '/train_labels.npy')
test = np.load(data_path + '/test.npy')
test_labels = np.load(data_path + '/test_labels.npy')

# classifier
pred_class_mahalanobis(train, train_labels, test)
'''
