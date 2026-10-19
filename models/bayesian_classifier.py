"""
Bayesian Gaussian Classifier for ASGSR MRI Classification

The classifier models the ASGSR statistical feature vector for each
class using a Gaussian distribution and computes posterior class
probabilities using Bayes' rule.

The posterior probabilities are also used to derive an
entropy-based confidence score.
"""

import numpy as np


class BayesianClassifier:
    """
    Gaussian Bayesian Classifier for ASGSR features.

    Model:
        p(f | c) = N(f ; mu_c, Sigma_c)

    Posterior:
        P(c | f) proportional to p(f | c) P(c)

    Confidence:
        C(f) = 1 - H(P) / log(|C|)

    Parameters
    ----------
    use_diagonal_cov : bool, optional
        If False, use the full covariance matrix.
        If True, use only the diagonal covariance terms.
    """

    def __init__(
        self,
        use_diagonal_cov=False,
        covariance_regularization=1e-6
    ):

        self.classes = None

        self.means = {}
        self.covariances = {}
        self.priors = {}

        self.use_diagonal_cov = use_diagonal_cov
        self.covariance_regularization = covariance_regularization

    # =================================================
    # TRAINING
    # =================================================

    def fit(self, X, y):
        """
        Fit the Gaussian Bayesian classifier.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (N, D).

        y : numpy.ndarray
            Class labels of shape (N,).

        Returns
        -------
        self
            Fitted classifier.
        """

        X = np.asarray(
            X,
            dtype=np.float64
        )

        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-D feature matrix of shape (N, D)."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1-D label vector."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        self.classes = np.unique(y)

        feature_dim = X.shape[1]

        for c in self.classes:

            # -----------------------------------------
            # Samples belonging to class c
            # -----------------------------------------

            X_c = X[y == c]

            if X_c.shape[0] == 0:
                continue

            # -----------------------------------------
            # Class mean
            # -----------------------------------------

            mean = np.mean(
                X_c,
                axis=0
            )

            # -----------------------------------------
            # Covariance estimation
            # -----------------------------------------

            if X_c.shape[0] > 1:

                cov = np.cov(
                    X_c,
                    rowvar=False
                )

            else:

                # Fallback for a single sample
                cov = np.eye(
                    feature_dim,
                    dtype=np.float64
                )

            # -----------------------------------------
            # Ensure covariance is a 2-D matrix
            # -----------------------------------------

            cov = np.asarray(
                cov,
                dtype=np.float64
            )

            if cov.ndim == 0:

                cov = np.array(
                    [[float(cov)]],
                    dtype=np.float64
                )

            elif cov.ndim == 1:

                cov = np.diag(
                    cov
                )

            # -----------------------------------------
            # Optional diagonal covariance
            # -----------------------------------------

            if self.use_diagonal_cov:

                cov = np.diag(
                    np.diag(cov)
                )

            # -----------------------------------------
            # Numerical covariance regularization
            # -----------------------------------------

            cov += (
                self.covariance_regularization
                * np.eye(
                    feature_dim
                )
            )

            # -----------------------------------------
            # Class prior probability
            # -----------------------------------------

            prior = (
                X_c.shape[0]
                / X.shape[0]
            )

            self.means[c] = mean
            self.covariances[c] = cov
            self.priors[c] = prior

        return self

    # =================================================
    # LOG GAUSSIAN LIKELIHOOD
    # =================================================

    def _log_gaussian_likelihood(
        self,
        x,
        mean,
        cov
    ):
        """
        Compute the log probability density of a
        multivariate Gaussian distribution.

        A log-domain implementation is used for numerical
        stability.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vector.

        mean : numpy.ndarray
            Class mean vector.

        cov : numpy.ndarray
            Class covariance matrix.

        Returns
        -------
        float
            Log Gaussian likelihood.
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        mean = np.asarray(
            mean,
            dtype=np.float64
        )

        cov = np.asarray(
            cov,
            dtype=np.float64
        )

        dimension = len(mean)

        # ---------------------------------------------
        # Additional covariance stabilization
        # ---------------------------------------------

        cov_stable = (
            cov
            + self.covariance_regularization
            * np.eye(dimension)
        )

        # ---------------------------------------------
        # Sign and logarithm of determinant
        # ---------------------------------------------

        sign, logdet = np.linalg.slogdet(
            cov_stable
        )

        # ---------------------------------------------
        # Fallback if covariance is numerically invalid
        # ---------------------------------------------

        if sign <= 0 or not np.isfinite(logdet):

            cov_stable = (
                cov_stable
                + 1e-5
                * np.eye(dimension)
            )

            sign, logdet = np.linalg.slogdet(
                cov_stable
            )

        if sign <= 0 or not np.isfinite(logdet):

            raise ValueError(
                "Covariance matrix is not positive definite."
            )

        # ---------------------------------------------
        # Solve covariance system
        #
        # Using solve is preferred over explicitly
        # computing a matrix inverse.
        # ---------------------------------------------

        diff = (
            x - mean
        )

        try:

            solved = np.linalg.solve(
                cov_stable,
                diff
            )

        except np.linalg.LinAlgError:

            # Safe fallback for ill-conditioned matrices
            inverse_cov = np.linalg.pinv(
                cov_stable
            )

            solved = (
                inverse_cov @ diff
            )

        # ---------------------------------------------
        # Mahalanobis quadratic term
        # ---------------------------------------------

        mahalanobis = np.dot(
            diff,
            solved
        )

        # ---------------------------------------------
        # Multivariate Gaussian log density
        # ---------------------------------------------

        log_normalizer = -0.5 * (
            dimension * np.log(
                2.0 * np.pi
            )
            + logdet
        )

        log_probability = (
            log_normalizer
            - 0.5 * mahalanobis
        )

        return float(
            log_probability
        )

    # =================================================
    # BACKWARD-COMPATIBLE GAUSSIAN LIKELIHOOD
    # =================================================

    def _gaussian_likelihood(
        self,
        x,
        mean,
        cov
    ):
        """
        Compute the Gaussian likelihood.

        This method is retained for compatibility with
        the original implementation.
        """

        log_probability = (
            self._log_gaussian_likelihood(
                x,
                mean,
                cov
            )
        )

        # Avoid numerical overflow/underflow where possible.
        log_probability = np.clip(
            log_probability,
            -745.0,
            700.0
        )

        return np.exp(
            log_probability
        )

    # =================================================
    # POSTERIOR PROBABILITY
    # =================================================

    def predict_proba(self, X):
        """
        Compute posterior class probabilities P(c | f).

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (N, D).

        Returns
        -------
        probabilities : numpy.ndarray
            Posterior probabilities of shape
            (N, number_of_classes).
        """

        X = np.asarray(
            X,
            dtype=np.float64
        )

        if X.ndim == 1:
            X = X.reshape(
                1,
                -1
            )

        if self.classes is None:
            raise RuntimeError(
                "The classifier must be fitted before prediction."
            )

        posterior_probabilities = []

        for x in X:

            log_posteriors = []

            for c in self.classes:

                # -------------------------------------
                # Gaussian log likelihood
                # -------------------------------------

                log_likelihood = (
                    self._log_gaussian_likelihood(
                        x,
                        self.means[c],
                        self.covariances[c]
                    )
                )

                # -------------------------------------
                # Bayesian prior
                # -------------------------------------

                log_prior = np.log(
                    self.priors[c]
                    + 1e-12
                )

                log_posteriors.append(
                    log_likelihood
                    + log_prior
                )

            log_posteriors = np.asarray(
                log_posteriors,
                dtype=np.float64
            )

            # -----------------------------------------
            # Stable softmax / log-sum-exp normalization
            # -----------------------------------------

            max_log = np.max(
                log_posteriors
            )

            probabilities = np.exp(
                log_posteriors
                - max_log
            )

            probability_sum = np.sum(
                probabilities
            )

            if (
                probability_sum <= 0
                or not np.isfinite(probability_sum)
            ):

                probabilities = np.ones(
                    len(self.classes),
                    dtype=np.float64
                )

                probabilities /= (
                    len(self.classes)
                )

            else:

                probabilities /= (
                    probability_sum
                )

            posterior_probabilities.append(
                probabilities
            )

        return np.asarray(
            posterior_probabilities,
            dtype=np.float64
        )

    # =================================================
    # PREDICTION
    # =================================================

    def predict(self, X):
        """
        Predict the class having the maximum posterior
        probability.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.

        Returns
        -------
        predictions : numpy.ndarray
            Predicted class labels.
        """

        probabilities = (
            self.predict_proba(X)
        )

        indices = np.argmax(
            probabilities,
            axis=1
        )

        return self.classes[
            indices
        ]

    # =================================================
    # ENTROPY-BASED CONFIDENCE
    # =================================================

    def compute_confidence(
        self,
        probs
    ):
        """
        Compute normalized entropy-based confidence.

        C(f) = 1 - H(P) / log(|C|)

        where

            H(P) = -sum_c P(c|f) log P(c|f)

        Parameters
        ----------
        probs : numpy.ndarray
            Posterior probability matrix of shape
            (N, number_of_classes).

        Returns
        -------
        confidence : numpy.ndarray
            Confidence values in [0, 1].
        """

        probs = np.asarray(
            probs,
            dtype=np.float64
        )

        if probs.ndim == 1:
            probs = probs.reshape(
                1,
                -1
            )

        eps = 1e-12

        num_classes = (
            probs.shape[1]
        )

        # ---------------------------------------------
        # Ensure valid probability values
        # ---------------------------------------------

        probs = np.clip(
            probs,
            eps,
            1.0
        )

        probs /= np.sum(
            probs,
            axis=1,
            keepdims=True
        )

        # ---------------------------------------------
        # Shannon entropy
        # ---------------------------------------------

        entropy = -np.sum(
            probs
            * np.log(probs),
            axis=1
        )

        # ---------------------------------------------
        # Maximum entropy for uniform distribution
        # ---------------------------------------------

        max_entropy = np.log(
            num_classes
        )

        # ---------------------------------------------
        # Normalized confidence
        # ---------------------------------------------

        if max_entropy > 0:

            confidence = (
                1.0
                - entropy / max_entropy
            )

        else:

            confidence = np.ones(
                probs.shape[0],
                dtype=np.float64
            )

        # ---------------------------------------------
        # Numerical clipping
        # ---------------------------------------------

        confidence = np.clip(
            confidence,
            0.0,
            1.0
        )

        return confidence

    # =================================================
    # PREDICT WITH CONFIDENCE
    # =================================================

    def predict_with_confidence(
        self,
        X
    ):
        """
        Predict class, posterior probability and
        entropy-based confidence.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.

        Returns
        -------
        predictions : numpy.ndarray
            Predicted class labels.

        confidence : numpy.ndarray
            Entropy-based confidence scores.

        probabilities : numpy.ndarray
            Posterior class probabilities.
        """

        probabilities = (
            self.predict_proba(X)
        )

        indices = np.argmax(
            probabilities,
            axis=1
        )

        predictions = self.classes[
            indices
        ]

        confidence = (
            self.compute_confidence(
                probabilities
            )
        )

        return (
            predictions,
            confidence,
            probabilities
        )
