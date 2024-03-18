# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        
        # Determining the Z-curr transformation matrix
        # Also transposing A_prev to (features, batch_size)
        Z_curr = np.dot(W_curr, A_prev.T) + b_curr

        # Checking for activation function selection

        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            raise ValueError("Activation function not found.")
        
        # Returning the activation matrix with transposition 
        return A_curr.T, Z_curr



    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        
        # Assigning input matrix
        A_curr = X

        # Cache to store intermediate values
        cache = {}

        # Adding input matrix to cache
        cache['A0'] = A_curr
      
        # Iterating through layers
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1         # Getting layer index
            
            # Retrieve the layer weight and bias matrix
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]

            # Get the activation function of layer
            activation = layer['activation']

            # Re-assign current matrix
            A_prev = A_curr

            # Forward pass through single current layer
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            
            # Store activation and linear transformation matrix in cache
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr


        return A_curr, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        # Retrieve Batch Size
        Batch_Size = A_prev.shape[0]

        # Check for activation function of the layer
        if activation_curr == 'relu':
            # Determine the derivative of ReLU activation function
            dZ_curr = self._relu_backprop(dA_curr.T, Z_curr)
        
        elif activation_curr == 'sigmoid':
            # Determine the derivative of sigmoid function
            dZ_curr = self._sigmoid_backprop(dA_curr.T, Z_curr)

        else:
            raise ValueError("Activation Function not found.")
        
        # Determine the gradients with respect to parameters
        dW_curr = np.dot(dZ_curr, A_prev) / Batch_Size
        
        # Determine gradient with respect to biases
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / Batch_Size

        # Determine gradient with respect to activations of previous layer
        dA_prev = np.dot(W_curr.T, dZ_curr).T

        return dA_prev, dW_curr, db_curr


    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        
        # Initialize gradient dictionary
        grad_dict = {}

        # Determine derivative of loss function with respect to predicted values
        if self._loss_func == 'binary_cross_entropy':
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_prev = self._mean_squared_error_backprop(y, y_hat)

        # Loop in reverse order for backpropogation
        for layer_idx in range(len(self.arch), 0, -1):
            # Retrieve cached values 
            A_prev = cache['A' + str(layer_idx - 1)]
            Z_curr = cache['Z' + str(layer_idx)]
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation_curr = self.arch[layer_idx - 1]['activation']

            # Compute gradients for the current layer
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_prev, activation_curr)
            
            # Store gradients 
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr
        
        return grad_dict
    
    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer_idx, _ in enumerate(self.arch, start=1):
            
            # Retrieve gradients for the current layer
            dW_curr = grad_dict['dW' + str(layer_idx)]
            db_curr = grad_dict['db' + str(layer_idx)]

            # Update weights and biases using gradient descent
            self._param_dict['W' + str(layer_idx)] -= self._lr * dW_curr 
            self._param_dict['b' + str(layer_idx)] -= self._lr * db_curr


    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        # Initializing empty lists for the outputs
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        
        y_train = y_train.reshape(-1, 1) # Reshaping to make 2D array
        y_val = y_val.reshape(-1, 1) # Reshaping to make 2D array

        # Iterating over epochs
        for _ in range(self._epochs):
            
            # Split training data into mini-batches
            num_batches = len(X_train) // self._batch_size
            
            for i in range(num_batches):
                # Get the batch sizes
                start_idx = i * self._batch_size
                end_idx = (i + 1) * self._batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
            
                # Forward pass on batch
                y_hat_batch, cache_batch = self.forward(X_batch)

                # Backward pass
                grad_dict = self.backprop(y_batch, y_hat_batch, cache_batch)

                # Update weights
                self._update_params(grad_dict)

            # Forward pass on training and test sets 
            y_hat_train, _ = self.forward(X_train)
            y_hat_val, _ = self.forward(X_val)

            # Computing losses
            if self._loss_func == 'binary_cross_entropy':
                loss_train = self._binary_cross_entropy(y_train, y_hat_train)
                loss_val = self._binary_cross_entropy(y_val, y_hat_val)

            if self._loss_func == 'mean_squared_error':
                loss_train = self._mean_squared_error(y_train, y_hat_train)
                loss_val = self._mean_squared_error(y_val, y_hat_val)

            per_epoch_loss_train.append(loss_train)
            per_epoch_loss_val.append(loss_val)

        return per_epoch_loss_train, per_epoch_loss_val       
            

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """

        # Executing forward pass
        y_hat, _ = self.forward(X)

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # Executing sigmoid function
        nl_transform = 1 / (1 + np.exp(-Z))

        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # Compute sigmoid derivative
        sigmoid_Z = 1 / (1 + np.exp(-Z))
        dZ = dA * sigmoid_Z * (1 - sigmoid_Z)
        
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # Compute ReLU activation function
        nl_transform = np.maximum(0, Z)
        
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # Compute ReLU derivative
        dZ = np.where(Z > 0, dA, 0)
        
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Clipping to prevent instable values
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        # Compute binary cross entropy loss
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Edge case for 1D array
        if y_hat.ndim == 1:
            m = 1
        else:
            m = y.shape[1]
            
        # Clipping to prevent instable values
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        # Compute binary cross entropy derivative
        dA = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat)) / m
        
        return dA


    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        # Compute mean squared error loss
        loss = np.mean(np.square(y - y_hat))
        
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Compute the derivative
        dA = (2 * (y_hat-y)) / len(y)

        return dA