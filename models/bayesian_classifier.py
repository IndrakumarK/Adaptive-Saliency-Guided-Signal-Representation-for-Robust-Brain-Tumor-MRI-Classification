import numpy as np


class BayesianClassifier:
    """
    Gaussian Bayesian Classifier for ASGSR features

    - Models p(f|c) using multivariate Gaussian
    - Computes posterior P(c|f)
    - Provides entropy-based confidence
    """

    def __init__(self):
        self.classes = None
        self.means = {}
        self.covariances = {}
        self.priors = {}

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------
    def fit(self, X, y):
        """
        X: (N, D) feature matrix
        y: (N,) labels
        """
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]

            # Mean
            mean = np.mean(X_c, axis=0)

            # Covariance with regularization
            cov = np.cov(X_c, rowvar=False)
            cov += 1e-6 * np.eye(cov.shape[0])

            # Prior probability
            prior = X_c.shape[0] / X.shape[0]

            self.means[c] = mean
            self.covariances[c] = cov
            self.priors[c] = prior

    # -------------------------------------------------
    # GAUSSIAN LIKELIHOOD (NUMERICALLY STABLE)
    # -------------------------------------------------
    def _gaussian_likelihood(self, x, mean, cov):
        """
        Stable multivariate Gaussian likelihood
        """
        dim = len(mean)
        eps = 1e-6

        # Stabilize covariance
        cov_stable = cov + eps * np.eye(dim)

        # Pseudo-inverse (safe)
        inv = np.linalg.pinv(cov_stable)

        # Stable determinant using slogdet
        sign, logdet = np.linalg.slogdet(cov_stable)

        if sign <= 0:
            logdet = np.log(np.abs(np.linalg.det(cov_stable)) + eps)

        # Compute log-likelihood
        diff = x - mean
        exponent = -0.5 * np.dot(np.dot(diff.T, inv), diff)

        log_norm = -0.5 * (dim * np.log(2 * np.pi) + logdet)

        log_prob = log_norm + exponent

        return np.exp(log_prob)

    # -------------------------------------------------
    # POSTERIOR PROBABILITY
    # -------------------------------------------------
    def predict_proba(self, X):
        """
        Returns P(c|f) for each sample
        """
        probs = []

        for x in X:
            class_probs = []

            for c in self.classes:
                likelihood = self._gaussian_likelihood(
                    x, self.means[c], self.covariances[c]
                )
                posterior = likelihood * self.priors[c]
                class_probs.append(posterior)

            class_probs = np.array(class_probs)

            # Normalize safely
            total = np.sum(class_probs) + 1e-8
            class_probs = class_probs / total

            probs.append(class_probs)

        return np.array(probs)

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    # -------------------------------------------------
    # CONFIDENCE (ENTROPY-BASED)
    # -------------------------------------------------
    def compute_confidence(self, probs):
        """
        C(f) = 1 - H(P) / log(|C|)
        """
        eps = 1e-8
        num_classes = probs.shape[1]

        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        max_entropy = np.log(num_classes)

        confidence = 1 - (entropy / (max_entropy + eps))
        return confidence

    # -------------------------------------------------
    # PREDICT WITH CONFIDENCE
    # -------------------------------------------------
    def predict_with_confidence(self, X):
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        conf = self.compute_confidence(probs)

        return preds, conf, probs