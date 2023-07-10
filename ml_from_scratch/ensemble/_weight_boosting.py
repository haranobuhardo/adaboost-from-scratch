import copy
import numpy as np
import time


class DecisionStump:
    """
    Decision Stump based class. Generally a decision tree with max-depth = 1

    Parameters
    ----------
    feature_index : int, default=None
        Feature to split a node

    threshold : float, default=None
        Threshold to split a node on a specific feature

    polarity : {1, -1}, default=1
        The polarity of stump prediction
        If, X[feature_index] > threshold, predict polarity
        Otherwise, predict the opposite of polarity

        The pair of polarity [polarity, opposite of polarity]
        - [1, -1]
        - [-1, 1]
    """
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        polarity=1
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.polarity = polarity

    def _best_split(self, X, y, weights):
        """
        Find the best split for a node
        
        Fundamental assumtion
            - y_pred where X[:, feature_i] > threshold = polarity
            - otherwise opposite of polarity
                - if polarity = 1, the opposite = -1
                - if polarity = -1, the opposite = 1
        """
        best_feature = None
        best_threshold = None
        best_polarity = None
        min_error = float('inf')

        for feature_i in range(self.n_features):
            # generate possible threshold
            ## sort the X and y based on X[feature_i] value
            idx = np.argsort(X[:, feature_i])
            X_i = X[idx, feature_i]
            y_i = y[idx]

            ## check the 'change' point of y, and store the index
            ## best threshold will only occur on a point where y value is changing
            changes = np.where(y_i[:-1] != y_i[1:])[0]

            ## The possible split points (threshold) would be the midpoints between 'change' points (between value of the index that y value is changing).
            ## this is simply just the midpoint of each two points
            midpoints = X_i[:-1] + np.diff(X_i) / 2

            ## iterate based on the 'change'
            for idx in changes:
                threshold = midpoints[idx]
                polarity = 1
                predictions = np.ones(self.n_samples) * polarity
                predictions[X_i <= threshold] = -1
                weighted_error = np.sum(weights[y_i != predictions])

                if weighted_error > 0.5:
                    weighted_error = 1 - weighted_error
                    polarity = -1

                if weighted_error < min_error:
                    best_polarity = polarity
                    best_threshold = threshold
                    best_feature = feature_i
                    min_error = weighted_error
                        
        return best_feature, best_threshold, best_polarity

    def fit(self, X, y, weights):
        """
        Fit a decision stump

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples

        y : {array-like} of shape (n_samples,)
            The output samples

        weights : {array-like} of shape (n_samples,)
            The samples weight
        """
        # Preparation
        X = np.array(X).copy()
        y = np.array(y).copy()
        self.n_samples, self.n_features = X.shape
        
        # Data check
        if len(np.unique(y)) > 2:
            raise Exception("this algorithm only support for binary classification at the moment")
        elif not np.array_equal(np.unique(y), np.array([-1, 1])):
            raise Exception("y class should be -1 and 1 only (please replace your data's class with this)")

        # Find the best split
        best_split_results = self._best_split(X, y, weights)

        # Extract results
        self.feature_index = best_split_results[0]
        self.threshold = best_split_results[1]
        self.polarity = best_split_results[2]

    def predict(self, X):
        """
        Predict the value with Decision Stump

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The sample input

        Return
        ------
        y : {array-like} of shape (n_samples,)
            The predicted value
        """
        # Convert input data
        X = np.array(X).copy()
        n_samples = X.shape[0]

        # Generate predictions
        y_pred = np.ones(n_samples) * self.polarity
        if self.polarity == 1:
            y_pred[X[:, self.feature_index] <= self.threshold] = -1
        else:
            y_pred[X[:, self.feature_index] <= self.threshold] = 1

        return y_pred

class AdaBoostClassifier:
    """
    AdaBoost Classifier

    Parameters
    ----------
    estimator : object, default=None
        The base estimator for Adaboost model
        If `None`, the base estimator would be the Decision stump

    n_estimators : int, default=5
        The number of boosting stage (also stump) 

    weighted_sampled : bool, default=False
        Choice to create a weighted sampled based on sample weights
        If True, sample with bigger probability will more likely to be sampled.
        Sampling is done with replacement
    """
    def __init__(
        self,
        estimator=None,
        n_estimators=5,
        weighted_sampled=False,
        random_state=42
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.weighted_sampled = weighted_sampled
        self.random_state = random_state

    def _generate_sample_indices(self, n_population, n_samples, weights):
        """
        Generate sample indices based on weights

        Parameters
        ----------
        n_population : int
            The number of maximum indices

        n_samples : int
            The number of sample to generate

        weights : {array-like} of (n_population,)
            The weights
        """
        # np.random.seed(self.random_state)
        
        sample_indices = np.random.choice(n_population, n_samples, p=weights)

        return sample_indices

    def fit(self, X, y):
        """
        Fit the Adaboost model

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input data

        y : {array-like} of shape (n_samples,)
            The output data
        """
        # Set seed for consistent result
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Check the base estimator
        if self.estimator is None:
            base_estimator = DecisionStump()

        # Convert input & output data
        X = np.array(X).copy()
        y = np.array(y).copy()
        self.n_samples, self.n_features = X.shape

        # Initialize
        self.weights = np.ones(self.n_samples) / self.n_samples
        self.alpha = np.zeros(self.n_estimators)
        self.estimators = []

        # Start training
        for i in range(self.n_estimators):
            # Generate the sample
            if self.weighted_sampled == False:
                sample_indices = self._generate_sample_indices(n_population = self.n_samples,
                                                          n_samples = self.n_samples,
                                                          weights = self.weights)
                X_ = X[sample_indices, :]
                y_ = y[sample_indices]
            else:
                X_ = X
                y_ = y
            
            # Create the estimator
            estimator = copy.deepcopy(base_estimator)

            # Train the estimator
            estimator.fit(X_, y_, weights = self.weights)

            # Predict & calculate the weighted error
            y_pred = estimator.predict(X_)
            weighted_error = np.sum(self.weights[y_ != y_pred])

            # Update the alpha
            error_odds = (1.0 - weighted_error) / (weighted_error + 1e-10)
            alpha = 0.5 * np.log(error_odds)

            # Update the weights
            self.weights *= np.exp(-alpha * y_ * y_pred)
            
            # Normalize the weights
            self.weights /= np.sum(self.weights)

            # Append the model & alpha
            self.estimators.append(estimator)
            self.alpha[i] = alpha
 
    def predict(self, X):
        """
        Predict the class

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : {array-like} of shape (n_samples,)
            The predicted data
        """
        # Convert the input
        X = np.array(X).copy()
        n_samples = X.shape[0]

        # Initialize (vector of output pred)
        y_pred = np.zeros(n_samples)
        
        # Predict using all trained weak-learners (estimators)
        for i in range(self.n_estimators):
            # Get the model
            estimator = self.estimators[i]
            alpha = self.alpha[i]

            # Predict
            predictions = estimator.predict(X)

            # Weighted sum the prediction
            y_pred += alpha * predictions

        # Do the sign function
        y_pred = np.sign(y_pred)
        
        return y_pred