import numpy as np


class PhotoGraph:
    def __init__(self, images, masks, alpha=1, beta=1):
        assert isinstance(images, np.ndarray)
        assert isinstance(masks, np.ndarray)
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

        self.pixels_costs = None
        self.transitions_costs = None
        self.total_costs = None
        self.mask_idxes = None

    def _get_pixels_costs(self):
        return self.alpha * (1 - self.masks)

    def _get_transitions_costs(self):
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
                    self.beta * (np.linalg.norm((self.images[from_image_id] - self.images[to_image_id]), ord=1, axis=2) +
                                 np.linalg.norm((shifted_images[from_image_id] - shifted_images[to_image_id]), ord=1, axis=2))

        return transitions_costs

    def _get_total_costs(self):
        total_costs = np.zeros((self.num_images, self.high, self.width))

        for n in range(self.high - 1, -1, -1):
            pixels_col_cost = self.pixels_costs[:, :, n]
            transitions_col_costs = self.transitions_costs[:, :, :, n]

            if n == self.high - 1:
                total_costs[:, :, n] = np.min(pixels_col_cost + transitions_col_costs, axis=1)
            else:
                total_costs[:, :, n] = np.min(pixels_col_cost + transitions_col_costs +
                                              total_costs[:, :, n + 1], axis=1)

        return total_costs

    def _get_mask_idxes(self):
        mask_idxes = np.zeros((self.high, self.width), dtype=int)

        for n in range(self.high):
            pixels_col_cost = self.pixels_costs[:, :, n]
            transitions_col_costs = self.transitions_costs[:, :, :, n]
            col_total_costs = self.total_costs[:, :, n]

            if n == 0:
                mask_idxes[:, n] = np.argmin(pixels_col_cost + col_total_costs, axis=0)

            else:
                for row_id, row_mask in enumerate(mask_idxes[:, n - 1]):
                    mask_idxes[row_id, n] = np.argmin(pixels_col_cost[:, row_id] +
                                                      transitions_col_costs[row_mask, :, row_id] +
                                                      col_total_costs[:, row_id], axis=0)

        return mask_idxes

    def merge_images(self):
        self.pixels_costs = self._get_pixels_costs()
        self.transitions_costs = self._get_transitions_costs()
        self.total_costs = self._get_total_costs()
        self.mask_idxes = self._get_mask_idxes()

        output_image = np.zeros_like(self.images[0])

        for row_id in range(self.high):
            for col_id in range(self.width):
                image_id = self.mask_idxes[row_id, col_id]
                output_image[row_id, col_id, :] = self.images[image_id, row_id, col_id, :]

        return output_image





