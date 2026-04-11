import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # Compute Y_hat = X @ W (matrix multiplication)
        # X is (n, 3), weights is (3,) -> result is (n,) predictions
        result = X @ weights
        return np.round(result, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute MSE = mean((predictions - truth)^2)
        # Use np.mean() and np.square()
        error = np.mean(np.square(model_prediction - ground_truth))
        return round(error, 5)