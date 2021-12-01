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

        self.field_labels = None
        self.dependencies = None

    def _get_initial_field_labels(self):
        field_labels = []

        for row_id in range(self.n):
            row_labels = []

            for col_id in range(self.n):
                field_value = self.field_array[row_id][col_id]
                field_label = []

                if field_value == 0:
                    field_label = list(range(1, self.n + 1))
                elif field_value > 0:
                    field_label = [field_value]
                row_labels.append(field_label)

            field_labels.append(row_labels)

        return np.array(field_labels)

    def _get_arcs(self):

        def get_row_arcs(field_labels, row_id):
            row_labels = field_labels[row_id]
            print(row_labels)
            return

        def get_column_arcs(field_labels, col_id):
            col_labels = field_labels[:, col_id]
            print(col_labels)
            return

        def get_block_arcs(field_labels, block_row_block, block_col_block, n):
            block_labels = field_labels[block_row_block * n: block_row_block * (n + 1), block_col_block * n: block_col_block * (n + 1),]
            print(block_labels)
            return

        def get_block_id(row_id, col_id, n):
            block_col_block = col_id // n
            block_row_block = row_id // n

            return block_row_block, block_col_block

        arcs = np.zeros_like(self.field_array)

        for row_id in range(self.n):
            for col_id in range(self.n):
                block_row_block, block_col_block = get_block_id(row_id, col_id, self.n)

                row_arcs = get_row_arcs(self.field_labels, row_id)
                column_arcs = get_column_arcs(self.field_labels, col_id)
                block_arcs = get_block_arcs(self.field_labels, block_row_block, block_col_block)

                running_arcs = row_arcs + column_arcs + block_arcs
                arcs[row_id][col_id] = running_arcs

        return arcs

    def prepare_to_ac(self):
        self.field_labels = self._get_initial_field_labels()
        self.arcs = self._get_arcs()

