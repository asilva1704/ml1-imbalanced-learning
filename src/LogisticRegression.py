import autograd.numpy as np
from autograd import grad

# MÉTRICAS
EPS = 1e-15

def binary_crossentropy(actual, predicted):
    predicted = np.clip(predicted, EPS, 1 - EPS)
    return np.mean(-np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))

def mean_squared_error(actual, predicted):
    return np.mean((actual - predicted) ** 2)

# CLASSE BASE
class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.size == 0:
            raise ValueError("Got an empty matrix.")
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missing required argument y")
            y = np.ravel(y)  # <- aqui está a correção
            self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()


# MODELOS DE REGRESSÃO
class BasicRegression(BaseEstimator):
    def __init__(self, lr=0.001, penalty="None", C=0.01, tolerance=0.0001, max_iters=1000):
        self.lr = lr
        self.penalty = penalty
        self.C = C
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.errors = []
        self.theta = []

    def _loss(self, w):
        raise NotImplementedError()

    def init_cost(self):
        raise NotImplementedError()

    def _add_penalty(self, loss, w):
        if self.penalty == "l1":
            loss += self.C * np.abs(w[1:]).sum()
        elif self.penalty == "l2":
            loss += 0.5 * self.C * (w[1:] ** 2).sum()
        return loss

    def _cost(self, X, y, theta):
        prediction = X.dot(theta)
        error = self.cost_func(y, prediction)
        return error

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = X.shape
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)
        self.X = self._add_intercept(self.X)
        self._train()

    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def _train(self):
        self.theta, self.errors = self._gradient_descent()

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return X.dot(self.theta)

    def _gradient_descent(self):
        theta = self.theta
        errors = [self._cost(self.X, self.y, theta)]
        cost_d = grad(self._loss)

        for i in range(1, self.max_iters + 1):
            delta = cost_d(theta)
            theta -= self.lr * delta
            errors.append(self._cost(self.X, self.y, theta))

            if np.linalg.norm(errors[i - 1] - errors[i]) < self.tolerance:
                break

        return theta, errors


class LogisticRegression(BasicRegression):
    def init_cost(self):
        self.cost_func = binary_crossentropy

    def _loss(self, w):
        loss = self.cost_func(self.y, self.sigmoid(np.dot(self.X, w)))
        return self._add_penalty(loss, w)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))  # Função sigmoide clássica

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

