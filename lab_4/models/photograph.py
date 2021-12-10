import numpy as np


class PhotoGraph:
    def __init__(self, images, masks, alpha=1, beta=1):
        assert type(images) == np.ndarray
        assert type(masks) == np.ndarray
        assert images.shape == masks.shape
        assert alpha >= 0
        assert beta >= 0

        self.high = images.shape[1]
        self.width = images.shape[2]
        self.num_images = len(images)
        self.images = images
        self.masks = masks
        self.alpha = alpha
        self.beta = beta
        self.images_pixels_costs = None
        self.transitions_costs = None

    def get_pixels_costs(self):
        return self.alpha * (1 - self.masks)

    def get_transitions_costs(self):
        def shift_right(images, shift):
            shifted_images = np.zeros_like(images)
            shifted_images[:, :, shift:] = images[:, :, shift:]

            return shifted_images

        shifted_images = shift_right(self.images, 1)

        # transitions_costs = np.zeros((row, from_pixel, from_image, to_image)
        transitions_costs = np.zeros((self.high, self.width, self.num_images, self.num_images))

        for from_image_id in range(self.num_images):
            for to_image_id in range(self.num_images):
                transitions_costs[:, :, from_image_id, to_image_id] = \
                    self.beta * (np.linalg.norm((images[from_image_id] - images[to_image_id]), ord=1, axis=2) +
                            np.linalg.norm((shifted_images[from_image_id] - shifted_images[to_image_id]), ord=1, axis=2))

        return transitions_costs



