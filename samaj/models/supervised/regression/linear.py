
   
# ******************************************************************************
# PROMPT
# ------
# Given a set of 2D data points, compute the polynomial that
# best fits the data.
# **************************************

from sklearn import preprocessing
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

from samaj.models.supervised import base

# ==============================================================================
# LOAD DATA
# ==============================================================================
# Load and separate the data
domain = np.arange(100)
noise = np.rand(100)
target = (2 * (domain ** 2)) + (5 * domain) + noise

# Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    domain, target, test_size=0.20, random_state=42
)

# ==============================================================================
# PROBLEM: Find the best fit polynomial
#
# APPROACH 1: Linear Regression - this is an example of NOT having enough
#                                 variance in the model
# ==============================================================================
class LinearRegressor(base.BaseModel):
    def __init__(
        self,
        evaluate_on={
            "MSE": make_scorer(mean_squared_error),
            "R2": make_scorer(r2_score),
        },
    ):
        super().__init__(evaluate_on)
        self.slope, self.intercept = 0, 0  # default values

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        """
        Approximates the slope and intercept using least-squares.
        This implementation borrows from Chapter 4 "Training Models"
        from Aurelion Geron's Hands-On Machine Learning (2019).
        Parameters:
        X_train(np.array): an (n x m) matrix, where
                            n = # of samples
                            m = # of features per sample
        y_train(np.array): 1D array of corresponding y-values
        Returns: None
        """
        # A: place the predictors in a matrix
        inputs = np.vstack([X_train.reshape(1, -1), np.ones(len(X_train))]).T
        # B: approximate the params of the line --> set instance attrs
        X_train_dagger = np.linalg.pinv(inputs)
        params = X_train_dagger.dot(y_train)
        self.slope, self.intercept = params

    def predict(self, X: np.array) -> np.array:
        """Estimates the corresponding y values for the given X.
        Parameter:
        X(np.array): an (n x m) matrix of predictor values
        Returns: 1D array of n predictions
        """
        inputs = np.vstack([X.reshape(1, -1), np.ones(len(X))]).T
        params = np.array([self.slope, self.intercept]).reshape(2, 1)
        return np.dot(inputs, params)  # dims are (n, 2) and (2, 1)


# ==============================================================================
# APPROACH 2: Polynomial Regression
# ==============================================================================
class PolynomialRegressor(LinearRegressor):
    def __init__(self):
        super().__init__()
        self.params = list()  # for however many params we need

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        """
        Approximates the slope and intercept using least-squares.
        Parameters:
        X_train(np.array): an (n x m) matrix, where
                            n = # of samples
                            m = # of features per sample
        y_train(np.array): 1D array of corresponding y-values
        Returns: None
        """
        # A: solve for the params of the function
        # params = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        X_train_dagger = np.linalg.pinv(X_train)
        params = X_train_dagger.dot(y_train)
        # B: set the params as instance attrs
        self.params = params

    def predict(self, X: np.array) -> np.array:
        """Estimates the corresponding y values for the given X.
        Parameter:
        X(np.array): an (n x m) matrix of predictor values
        Returns: 1D array of n predictions
        """
        params = self.params.reshape(X.shape[1], 1)
        return np.dot(X, params)


if __name__ == "__main__":
    num_cross_val_folds = 5
    print("==========================================================")
    print("Example: QUADRATIC REGRESSION", end="\n\n")
    print(
        f"training/testing the model with {num_cross_val_folds}-fold cross validation...",
        end="\n\n",
    )
    PolynomialRegressor.fit_evaluate(
        domain.reshape(-1, 1),
        target,
        logging=True,
        preprocessing=[preprocessing.PolynomialFeatures(2, include_bias=False)],
    )
