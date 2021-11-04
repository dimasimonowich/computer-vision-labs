import numpy as np

X_sample = np.array([[1.5, 2.2], [3.1, 5.5]])

class KozModel:
    def __init__(self, max_num_epoch=100, eps=0.01):
        self.max_num_epoch = max_num_epoch
        self.eps = eps
        self.beta = None

    def init_with_first_vec(self, X_vec):
        assert len(X_vec) > 0
        self.beta = X_vec[0]

    def load_points_to_vec(self, X):
        X_vec = []

        for x in X:
            assert len(x) == 2
            vec = [1, x[0], x[1], x[0]*x[0], x[0]*x[1], x[1]*x[0], x[1]*x[1]]
            X_vec.append(vec)

        return np.array(X_vec)

    def create_dummy_vecs(self):
        dummy_vecs = []
        sigma = np.array([[self.beta[3]], [self.beta[4]],
                          [self.beta[5]], [self.beta[6]]])

        lamdas, v = np.linalg.eig(sigma)

        for i in range(len(lamdas)):
            if lamdas[i] >= 0:
                dummy_vec = [0, 0, 0, v[i][0]*v[i][0], v[i][0]*v[i][1], v[i][1]*v[i][0], v[i][1]*v[i][1]]
                dummy_vecs.append(dummy_vec)

        return dummy_vecs

    def backward(self, vec, k):
        numerator = np.dot(vec, vec) - k * np.dot(vec, self.beta)
        denominator = np.dot(self.beta - k*vec, self.beta - k*vec)
        gamma = numerator/denominator
        self.beta = gamma*self.beta + (1 - gamma)*k*vec


    def fit(self, X, y):
        X_vec = self.load_points_to_vec(X)
        self.init_with_first_vec(X_vec)

        for epoch in range(self.max_num_epoch):
            running_beta = self.beta

            for m in range(len(X_vec)):
                logit = y[m] * np.dot(X_vec[m], self.beta)

                if logit <= 0:
                    self.backward(X_vec[m], y[m])
                    break

            # dummy_vecs = self.create_dummy_vecs()
            #
            # for dummy_vec in dummy_vecs:
            #     dummy_logit = 1 * np.dot(dummy_vec, self.beta)
            #
            #     if dummy_logit <= 0:
            #         self.backward(dummy_vec, 1)

            if np.sum(np.abs(running_beta - self.beta)) < self.eps:
                break

    def predict(self, X):
        X_vec = self.load_points_to_vec(X)
        prediction = []

        for vec in X_vec:
            logit = np.dot(vec, self.beta)

            if logit > 0:
                prediction.append(1)
            elif logit <= 0:
                prediction.append(-1)

        return prediction

    def fit_predict(self, X, y):
        self.fit(X, y)
        prediction = self.predict(X)

        return prediction