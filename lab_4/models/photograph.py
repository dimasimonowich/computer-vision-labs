import numpy as np


class PhotoGraph:
    def __init__(self, images, masks, alpha=1, beta=1):
        assert type(images) == np.ndarray
        assert type(masks) == np.ndarray
        assert images.shape == masks.shape
        assert alpha >= 0
        assert beta >= 0

        self.K = len(images)
        self.images = images
        self.masks = masks
        self.alpha = alpha
        self.images_pixels_costs = None
        self.transitions_costs = None

    def get_pixels_costs(self):
        return self.alpha * (1 - self.masks)


