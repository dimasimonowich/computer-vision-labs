import numpy as np

X_sample = np.array([[1.5, 2.2], [3.1, 5.5]])

class KozModel:
    def __init__(self, max_num_epoch=100):
        self.max_num_epoch = max_num_epoch
        self.beta = None

    def init_with_first_vec(self, X_vec):
        assert len(X_vec) > 0
        self.beta = X_vec[0]

    def check_fictitious_vector(self, sigma):
        pass

    def load_points_to_vec(self, X):
        assert len(X) == 2
        X_vec = []

        for x in X:
            vec = [1, x[0], x[1], x[0]*x[0], x[0]*x[1], x[1]*x[0], x[1]*x[1]]
            X_vec.append(vec)

        return X_vec

    def fit(self, X):
        X_vec = self.load_points_to_vec(X)
        self.init_with_first_point(X_vec)

        for epoch in range(self.max_num_epoch):

            for sample in X:
                pass

    def predict(self, y):
        pass
