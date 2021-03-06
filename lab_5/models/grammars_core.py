import numpy as np
from matplotlib import pyplot as plt


class Grammars:
    def __init__(self, train_data, rules, objects):
        self.train_data = train_data
        self.rules = rules
        self.objects = objects

    def _classify(self, image):
        predicted_class = None

        for data_tuple in self.train_data:
            train_images = data_tuple[0]
            label = data_tuple[1]

            for train_image in train_images:
                image_diag = np.diagonal(image)
                train_image_diag = np.diagonal(train_image)

                if np.all(image_diag == train_image_diag):
                    predicted_class = label
                    return predicted_class

        print("ERROR: Image can not be classified!")
        return predicted_class

    def _get_atoms(self, image):
        def split_image(image, chunk_shape):
            assert image.shape[0] % chunk_shape[0] == 0
            assert image.shape[1] % chunk_shape[1] == 0

            num_h_blocks = image.shape[0] / chunk_shape[0]
            num_v_blocks = image.shape[1] / chunk_shape[1]

            horizontal_blocks = np.array_split(image, num_h_blocks)
            chunks = [np.array_split(block, num_v_blocks, axis=1) for block in horizontal_blocks]

            return np.array(chunks)

        vertical_shapes = []
        horizontal_shapes = []

        for train_pair in self.train_data:
            vertical_shapes.append(train_pair[0].shape[1])
            horizontal_shapes.append(train_pair[0].shape[2])

        atom_shape = (min(vertical_shapes), min(horizontal_shapes))
        atoms = split_image(image, atom_shape)

        return atoms

    def get_initial_markup(self, image):
        atoms = self._get_atoms(image)

        initial_markup = np.empty(atoms.shape[:2], dtype=object)

        for row_id, atom_row in enumerate(atoms):
            for col_id, atom in enumerate(atom_row):
                atom_class = self._classify(atom)

                if atom_class:
                    initial_markup[row_id][col_id] = list(str(atom_class))
                else:
                    print("ERROR: Image can not be marked up!")
                    return None

        return initial_markup






