import numpy as np

from modules.metrics import *
from modules.utils import z_normalize


default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05}
                         }

class TimeSeriesKNN:
    """
    KNN Time Series Classifier

    Parameters
    ----------
    n_neighbors: number of neighbors
    metric: distance measure between time series
             Options: {euclidean, dtw}
    metric_params: dictionary containing parameters for the distance metric being used
    """
    
    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean', metric_params: dict | None = None) -> None:

        self.n_neighbors: int = n_neighbors
        self.metric: str = metric
        self.metric_params: dict | None = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)


    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> Self:
        """
        Fit the model using X_train as training data and Y_train as labels

        Parameters
        ----------
        X_train: train set with shape (ts_number, ts_length)
        Y_train: labels of the train set
        
        Returns
        -------
        self: the fitted model
        """
       
        self.X_train = X_train
        self.Y_train = Y_train

        return self


    def _distance(self, x_train, x_test):
        """
        Compute distance between the train and test samples
        
        Parameters
        ----------
        x_train: sample of the train set
        x_test: sample of the test set
        
        Returns
        -------
        dist: distance between the train and test samples
        """

        if self.metric=='euclidean':
            if self.metric_params['normalize']:
                dist_func = norm_ED_distance
            else:
                dist_func = ED_distance
        elif self.metric=='dtw':
            if self.metric_params['normalize']:
                x_train = z_normalize(x_train)
                x_test = z_normalize(x_test)
            dist_func = DTW_distance
        else:
            raise ValueError("Metric must be 'euclidean' or 'dtw'.")

        dist = dist_func(x_train, x_test, *list(self.metric_params.values())[1:])

        return dist



    def _find_neighbors(self, x_test):
        """
        Find the k nearest neighbors of the test sample

        Parameters
        ----------
        x_test: sample of the test set
        
        Returns
        -------
        neighbors: k nearest neighbors (distance between neighbor and test sample, neighbor label) for test sample
        """

        neighbors = []
        for i, x_train in enumerate(self.X_train):
            neighbors.append((self._distance(x_train, x_test), self.Y_train[i]))
        neighbors = sorted(neighbors, key=lambda x: x[0])[:self.n_neighbors]

        return neighbors


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for samples of the test set

        Parameters
        ----------
        X_test: test set with shape (ts_number, ts_length))

        Returns
        -------
        y_pred: class labels for each data sample from test set
        """

        from ipywidgets import IntProgress
        from IPython.display import display

        bar = IntProgress(min=0, max=len(X_test))
        display(bar)

        y_pred = []

        for x_test in X_test:
            neighbors = self._find_neighbors(x_test)
            labels = np.array(neighbors)[:, 1]
            unique_labels, counts = np.unique(labels, return_counts=True)
            y_pred.append(int(unique_labels[np.argmax(counts)]))
            bar.value += 1

        return y_pred


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy classification score

    Parameters
    ----------
    y_true: ground truth (correct) labels
    y_pred: predicted labels returned by a classifier

    Returns
    -------
    score: accuracy classification score
    """

    score = 0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            score = score + 1
    score = score/len(y_true)

    return score
