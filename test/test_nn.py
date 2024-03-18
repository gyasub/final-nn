# TODO: import dependencies and write unit tests below

# Imports 
import numpy as np
import pytest
from nn.nn import NeuralNetwork
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from nn.preprocess import one_hot_encode_seqs, sample_seqs
import random
from collections import Counter

# Defining the architecture for test autoencoder
autoencoder_architecture = [
    {'input_dim': 4, 'output_dim': 2, 'activation': 'relu'},
    {'input_dim': 2, 'output_dim': 8, 'activation': 'relu'}
]

# Define hyperparameters
learning_rate = 0.001
random_seed = 42
epochs = 100
batch_size = 10
loss_function = 'mean_squared_error'
#loss_function = 'binary_cross_entropy'

# Create an instance of NeuralNetwork for the autoencoder
autoencoder = NeuralNetwork(
    nn_arch=autoencoder_architecture,
    lr=learning_rate,
    seed=random_seed,
    batch_size=batch_size,
    epochs=epochs,
    loss_function=loss_function
)

# Creating input matrix
X = np.random.rand(batch_size, 4)

# Forward pass using my code
output, cache = autoencoder.forward(X)


def test_single_forward():
    # Create example weight, input and bias matrix
    W = np.array([[0.1, 0.2, 0.3],
     [0.4, 0.5, 0.6]])
    
    # Need to transpose A because expected input is 2x3
    A = np.array([[0.7, 0.8],
     [0.9, 0.1],
     [0.2, 0.3]])
    # Transpose
    A = A.T

    B = np.array([[0.1],
     [0.2]])    
    
    Expected_Z = np.array([[0.41, 0.29],
                  [1.05, 0.75]])
    # Running single forward pass
    A_curr, Z_curr = autoencoder._single_forward(W, B, A, 'relu')

    # Assert that the Z matrix has correct shape and values
    assert np.allclose(Z_curr, Expected_Z, atol=0.01)
 
def test_forward():
    
    # Assert if the output has the expected shape
    assert output.shape == (batch_size, 8)

    # Assert that cache has 3 A matrices and 2 Z matrix
    assert len(cache) == 5

def test_single_backprop():
    # Example set
    W_curr = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]])
    b_curr = np.array([0.1, 0.2]).reshape(-1, 1)
    Z_curr = np.array([[0.1, 0.2],
                       [0.3, 0.4]])
    A_prev = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]])
    dA_curr = np.array([[0.1, 0.2],
                        [0.3, 0.4]])
    activation_curr = 'relu'

    # Running my method
    dA_prev, dW_curr, db_curr = autoencoder._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

    # Expected outputs
    expected_dA_prev = np.array([[0.09, 0.12, 0.15],
                                  [0.19, 0.26, 0.33]])
    expected_dW_curr = np.array([[0.065, 0.085, 0.105],
                                  [0.09, 0.12, 0.15]])
    expected_db_curr = np.array([[0.2],
                                 [0.3]])
    
    # Assert that the computed derivatives match the expected output
    assert np.allclose(dA_prev, expected_dA_prev)
    assert np.allclose(dW_curr, expected_dW_curr)
    assert np.allclose(db_curr, expected_db_curr)

def test_predict():
    
    # Run predict method
    y_hat = autoencoder.predict(X)

    # Assert that output is expected
    assert np.allclose(y_hat, output)

def test_binary_cross_entropy():
    
    # Create test example
    y_true = np.array([1])
    y_pred = np.array([0.9])

    # Run the method
    loss = autoencoder._binary_cross_entropy(y_true, y_pred)

    # Assert that loss value is expected 
    assert np.isclose(loss, 0.105, atol=0.01)

    # 2nd test
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])
    loss = autoencoder._binary_cross_entropy(y_true, y_pred)
    
    # Assert that loss value is expected 
    assert np.isclose(loss, 0.144, atol= 0.01)

def test_binary_cross_entropy_backprop():
    # Create example
    y_true = np.array([1])
    y_pred = np.array([0.9])

    # Run the method
    dA = autoencoder._binary_cross_entropy_backprop(y_true, y_pred)
    
    # Assert expected value
    assert np.isclose(dA, -1.11, atol=0.01)

def test_mean_squared_error():
    
    # Define test data
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])
    
    # Call the method being tested
    mse_method = autoencoder._mean_squared_error(y_true, y_pred)
    
    # Compute the expected loss using scikit-learn
    mse_sklearn = mean_squared_error(y_true, y_pred)
    
    # Assert that the computed loss matches the expected loss
    assert np.isclose(mse_method, mse_sklearn)

def test_mean_squared_error_backprop():
    
    # Define test data
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])
    
    # Call the method being tested
    mse_backprop = autoencoder._mean_squared_error_backprop(y_true, y_pred)

    # Correct output
    expected_msearray = np.array([-0.06666667,  0.06666667, -0.13333333])

    # Assert that formula is computed correctly
    assert np.allclose(mse_backprop, expected_msearray)

def test_sample_seqs():
    
    # Define test data
    seqs = ['AGG', 'AGA', 'TCA', 'ATC', 'ACA', 'ATT']
    labels = [True, False, False, True, False, False]  # Assuming True represents the minority class

    # Setting seed
    random.seed(42)

    # Call the function
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # Count the number of sequences in each class
    label_counter = Counter(sampled_labels)

    # Assert that the total number of sampled sequences is correct
    assert len(sampled_seqs) == len(sampled_labels) == len(labels) + 2

    # Assert balanced classes
    assert label_counter[True] == label_counter[False]

    # Assert correct shuffling, ensuring all seqs are there
    assert set(sampled_seqs) == set(seqs)  
    assert set(sampled_labels) == set(labels) 


def test_one_hot_encode_seqs():
    
    # Create test seq
    seq = ['AGA']

    # Get output of the function
    encoded_array = one_hot_encode_seqs(seq)
    expected_array = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    
    # Assert that output is the expected array
    assert np.allclose(encoded_array, expected_array)