import numpy as np


class PhotoGraph:
    def __init__(self, images, masks, alpha=1, beta=1):
        assert type(images) == np.ndarray
        assert type(masks) == np.ndarray
        assert images[:, :, :, 0].shape == masks.shape
        assert alpha >= 0
        assert beta >= 0

        self.high = images.shape[1]
        self.width = images.shape[2]
        self.num_images = len(images)
        self.images = images
        self.masks = masks
        self.alpha = alpha
        self.beta = beta
        self.pixels_costs = self.get_pixels_costs()
        self.transitions_costs = self.get_transitions_costs()
        self.total_costs = self.get_total_costs()

    def get_pixels_costs(self):
        return self.alpha * (1 - self.masks)

    def get_transitions_costs(self):
        def shift_right(images, shift):
            shifted_images = np.zeros_like(images)
            shifted_images[:, :, shift:] = images[:, :, shift:]

            return shifted_images

        shifted_images = shift_right(self.images, 1)

        # transitions_costs = np.zeros((row, from_pixel, from_image, to_image)
        transitions_costs = np.zeros((self.num_images, self.num_images, self.high, self.width))

        for from_image_id in range(self.num_images):
            for to_image_id in range(self.num_images):
                transitions_costs[from_image_id, to_image_id, :, :] = \
                    self.beta * (np.linalg.norm((self.images[from_image_id] - self.images[to_image_id]), ord=2, axis=2) +
                                 np.linalg.norm((shifted_images[from_image_id] - shifted_images[to_image_id]), ord=2, axis=2))

        return transitions_costs

    def get_total_costs(self):
        total_costs = np.zeros((self.num_images, self.high, self.width))

        for n in range(self.high - 1, 0, -1):
            pixels_col_cost = self.pixels_costs[:, :, n]
            transitions_col_costs = self.transitions_costs[:, :, :, n]

            if n == self.high - 1:
                total_costs[:, :, n] = np.min(pixels_col_cost + transitions_col_costs, axis=1)
            else:
                total_costs[:, :, n] = np.min(pixels_col_cost + transitions_col_costs +
                                              total_costs[:, :, n + 1], axis=1)

        return total_costs

    def get_mask_idxes(self):
        mask_idxes = np.zeros((self.high, self.width), dtype=int)

        for n in range(self.high):
            pixels_col_cost = self.pixels_costs[:, :, n]
            transitions_col_costs = self.transitions_costs[:, :, :, n]
            col_total_costs = self.total_costs[:, :, n]

            if n == 0:
                mask_idxes[:, n] = np.argmin(pixels_col_cost + col_total_costs, axis=0)

            else:
                mask_idxes[:, n] = np.argmin(pixels_col_cost +
                                             transitions_col_costs[mask_idxes[:, n - 1], :, :][0] +
                                             col_total_costs, axis=0)

        return mask_idxes




