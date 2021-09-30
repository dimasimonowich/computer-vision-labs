import numpy as np


class EM():
    def __init__(self, eps=0.0001, num_epoch=10, num_classes=2):
        self.eps = eps
        self.num_epoch = num_epoch
        self.num_classes = num_classes
        self.p_ij_classes = None
        self.p_classes = None

    def initialize_probs(self, X):
        self.p_classes_Xm = np.random.uniform(0, 1, (self.num_classes, X.shape[0]))

    def perform_expectation_step(self, X):
        def get_p_target_class_Xm(p_ij_classes, p_classes, target_class, X):
            p_ij_classes = np.clip(p_ij_classes, self.eps, 1 - self.eps)
            # p_classes = (2, 1)
            # p_ij_classes = (2, 784)
            # class_idx = int
            # X = (14780, 784)

            all_classes = range(p_ij_classes.shape[0])
            non_target_classes = np.delete(all_classes, target_class)

            non_target_sum = 0

            for non_target_class in non_target_classes:
                p_ij_classes_ratio = (p_ij_classes[non_target_class] /
                                      p_ij_classes[target_class]) ** X
                # p_ij_classes_ratio = (14780, 784)

                reversed_p_ij_classes_ratio = ((1 - p_ij_classes[non_target_class]) /
                                               (1 - p_ij_classes[target_class])) ** (1 - X)
                # reversed_p_ij_classes_ratio = (14780, 784)

                p_classes_ratio = p_classes[non_target_class] / p_classes[target_class]
                # p_classes_ratio = float

                pixel_probs_product = np.prod(p_ij_classes_ratio * reversed_p_ij_classes_ratio, axis=1, keepdims=False)
                non_target_sum += p_classes_ratio * pixel_probs_product

            denominator = 1 + non_target_sum
            numerator = 1

            p_target_class_Xm = numerator / denominator
            # p_target_class_Xm = (14780, 1)

            return p_target_class_Xm

        for target_class in range(self.p_classes_Xm.shape[0]):
            self.p_classes_Xm[target_class] = get_p_target_class_Xm(self.p_ij_classes, self.p_classes, target_class, X)

    def perform_maximization_step(self, X):
        # p_classes_Xm = (2, 14780)
        # X = (14780, 784)

        denominator = np.sum(self.p_classes_Xm, axis=1, keepdims=True)
        # denominator = (2, 1)

        numerator = self.p_classes_Xm @ X
        # numerator = (2, 784)

        self.p_ij_classes = numerator / denominator
        # p_ij_classes = (2, 784)

    def update_p_classes(self, X):
        # p_classes_Xm = (2, 14780)
        # X = (14780, 784)

        num_images = X.shape[0]
        # num_images = 14780

        numerator = np.sum(self.p_classes_Xm, axis=1, keepdims=True)
        # numerator = (2, 1)

        self.p_classes = numerator / num_images
        # p_classes = (2, 1)

    def fit(self, X):
        # Initialization
        self.initialize_probs(X)

        for _ in range(self.num_epoch):
            # Maximization
            self.perform_maximization_step(X)
            self.update_p_classes(X)

            # Expectation
            self.perform_expectation_step(X)

    def fit_predict(self, X):
        self.fit(X)

        self.perform_expectation_step(X)
        prediction = self.p_classes_Xm.argmax(axis=0)

        return prediction

