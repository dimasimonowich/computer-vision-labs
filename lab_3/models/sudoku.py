import numpy as np


class Sudoku:
    def __init__(self, field):
        self.field_array = np.array(field)
        self.n = self.field_array.shape[0]

        assert self.field_array.shape[0] == self.field_array.shape[1]
        assert self.n >= 4
        assert np.all(self.field_array >= 0)
        assert np.all(self.field_array <= self.n)
        # TODO: assert all elements are ints

        self.field_labels = self._get_initial_field_labels()

    def _get_initial_field_labels(self):
        field_labels = []

        for row_id in range(self.n):
            row_labels = []

            for col_id in range(self.n):
                field_value = self.field_array[row_id][col_id]

                if field_value == 0:
                    field_label = list(range(1, self.n + 1))
                elif field_value > 0:
                    field_label = [field_value]
                row_labels.append(field_label)

            field_labels.append(row_labels)

        return np.array(field_labels)


