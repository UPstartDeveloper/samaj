import numpy as np
from sklearn.metrics import accuracy_score, make_scorer

from samaj.models.supervised import base


class BinarySVC(base.BaseModel):
    """
    Credit to this presentation by Patrick Loeber
    for helping me on the code: 
    https://www.youtube.com/watch?v=UX0f9BNBcsY
    """

    def __init__(
        self,
        learning_rate=0.001,
        lambda_param=0.01,
        epochs=1000,
        evaluate_on={"Accuracy": accuracy_score},
    ):
        super().__init__(evaluate_on)
        self.lr = learning_rate  # used for optimization
        self.regularizer = lambda_param  # used for regularization
        self.epochs = epochs
        self.w = None
        self.b = None

    def _linear_output(self, X):
        return np.dot(X, self.w) - self.b

    def fit(self, X_train, y_train):
        _, n_features = X_train.shape

        binary_labels = np.where(y_train <= 0, -1, 1)

        # init the params of the model
        self.w = np.zeros(n_features)
        self.b = 0

        # using gradient descent
        for _ in range(self.epochs):
            for index, sample in enumerate(X_train):
                is_positive_class = (
                    binary_labels[index] * (self._linear_output(sample)) >= 1
                )

                # derivatives used to update the weights
                if is_positive_class is True:
                    self.w -= self.lr * (2 * self.regularizer * self.w)
                else:  # belongs to the negative class
                    self.w -= self.lr * (
                        2 * self.regularizer * self.w
                        - np.dot(sample.reshape(-1, 1), binary_labels[index])
                    )
                    self.b -= self.lr * binary_labels[index]

    def predict(self, X):
        """For the purpose of binary classification, we only output +1 or -1."""
        return np.sign(self._linear_output(X))


class OneVersusRestSVC(base.BaseModel):
    """
    An ensemble approach for support vector classifiers 
    to handle multinomial datasets.
    """

    def __init__(
        self,
        learning_rate=0.001,
        lambda_param=0.01,
        epochs=1000,
        evaluate_on={"Accuracy": accuracy_score},
    ):
        super().__init__(evaluate_on)
        self.lr = learning_rate  # used for optimization
        self.regularizer = lambda_param  # used for regularization
        self.epochs = epochs
        self.classifiers = dict()

    def fit(self, X_train, y_train):
        # A: for each class, train a classifier
        labels = np.unique(y_train)
        for positive_label in labels:
            # transform the n-1 "non-positive" labels to all be -1
            y_train_transformed = np.where(y_train != positive_label, -1, 1)
            binary_model = BinarySVC()
            binary_model.fit(X_train, y_train_transformed)
            # save this trained model for later predictions
            self.classifiers[positive_label] = binary_model

    def predict(self, X):
        # get all the models to "vote" on if the sample(s) fall in their positive class
        votes = dict()
        for pos_label, model in self.classifiers.items():
            votes[pos_label] = model._linear_output(X)

        # decide what final label ought to be
        y_pred_final = np.zeros(X.shape[0])

        for index in range(y_pred_final.shape[0]):
            # linear search for the most confident prediction
            strongest_pred_confidence, pos_label_of_strongest = -1, None
            for pos_label, predictions in votes.items():
                model_pred = predictions[index]
                if abs(model_pred) > strongest_pred_confidence:
                    # strongest_pred = model_pred
                    pos_label_of_strongest = pos_label
            # now, we see if we can actually use that model's positive class label, or go to the others
            if model_pred > 0:
                y_pred_final[index] = pos_label_of_strongest
            else:  # model_pred < 0
                # need to take the positive label of one of the other classifiers
                # candidate_classes = [
                #     pos_label for pos_label in self.classifiers.keys()
                #     if pos_label != pos_label_of_strongest
                # ]
                # another search, narrowed down to JUST these classes
                labels_and_preds = [
                    (pos_label, predictions[index])
                    for pos_label, predictions in votes.items()
                    if pos_label != pos_label_of_strongest
                ]
                # take the max of these
                labels_and_preds.sort(
                    reverse=True, key=lambda label_pred_tuple: label_pred_tuple[1]
                )
                most_likely_label, _ = labels_and_preds[0]
                y_pred_final[index] = most_likely_label

        # all done!
        return y_pred_final
