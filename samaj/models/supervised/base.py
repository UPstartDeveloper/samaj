from typing import Callable, Dict

import numpy as np
from sklearn import model_selection
from sklearn.pipeline import make_pipeline


class BaseModel:
    """
    Common API across regressors.
    Subclasses need to implement "fit()" and "predict()"
    based on their specific needs.
    """

    def __init__(self, evaulate_on: Dict[str, Callable]):
        """
        Constructs an instance.

        Parameter:
            evaluate_on(dict): these are the metrics used to score the model,
                            where the keys are the name of the metric,
                            and the values are the corresponding scoring function
        """
        self.evaluate_on = evaulate_on

    @classmethod
    def fit_evaluate(
        cls,
        X: np.array,
        y: np.array,
        num_cross_val_folds=5,
        preprocessing=list(),
        logging=False,
    ) -> Dict[str, np.array]:
        """
        Builds, trains, and evaluates a data pipeline with this model.
        K-fold cross validation is used to give a better picture
        of model performance. Fitting time is also displayed.
        
        Parameters:
        X(np.array): an (n x m) matrix, where
                     n = # of samples
                     m = # of features per sample
        y(np.array): 1D array of corresponding y-values
        num_cross_val_folds(int): number of splits of data to make
        preprocessing(list): an array of any data preprocessing steps
                             to include in the pipeline.
                             Any instances of sklearn.preprocessing
                             classes will work great here.
        logging(bool): whether or not to print the performance/timing metrics
        
        Example Usage:
            SomeSubclass.fit_evaluate(X, y, preprocessing=[preprocessing.StandardScaler()])
        
        Returns: the output of sklearn.model_selection.cross_validate:
                 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        """
        # A: save a ref to a estimator instance
        estimator = cls()
        # B: build a pipeline
        steps = preprocessing[:]
        steps.append(estimator)
        pipeline = make_pipeline(*steps)
        # C: run cross validation (using the pipeline)
        scores = model_selection.cross_validate(
            pipeline,
            X,
            y,
            scoring=estimator.evaluate_on,
            cv=num_cross_val_folds,
            return_estimator=True,
        )
        # D: show resulting performance metrics (using the estimator)
        if logging:
            print(f"Average fitting time: {scores['fit_time'].mean()} s")
        for metric in estimator.evaluate_on.keys():
            # calculate the mean of this metric
            score_key = f"test_{metric}"
            avg = scores[f"test_{metric}"].mean()
            # log it
            if logging:
                print(f"Average {metric}: {avg}")
            # save it back in the dict for later use
            scores[f"{score_key}_mean"] = avg
        return scores
