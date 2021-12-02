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

        self.fields = None
        self.dependencies = None

    def _get_initial_fields_filling(self):
        def get_block_ids(row_id, col_id, n):
            block_col_id = col_id // (n**0.5)
            block_row_id = row_id // (n**0.5)

            return int(block_row_id), int(block_col_id)

        fields = []

        for row_id in range(self.n):
            row_fields = []

            for col_id in range(self.n):
                field = {}

                field_value = self.field_array[row_id][col_id]
                field_label = []

                if field_value == 0:
                    field_label = list(range(1, self.n + 1))
                elif field_value > 0:
                    field_label = [field_value]

                field['row_id'] = row_id
                field['col_id'] = col_id

                block_row_id, block_col_id = get_block_ids(row_id, col_id, self.n)
                field['block_row_id'] = block_row_id
                field['block_col_id'] = block_col_id

                field['labels'] = field_label

                row_fields.append(field)

            fields.append(row_fields)

        return fields

    def _get_arcs(self):

        def get_row_arcs(field_labels, row_id):
            row_labels = field_labels[row_id]
            row_arcs = {}

            for i, labels in enumerate(row_labels):
                for j, label in enumerate(labels):

                    row_arcs[f"{i}, {j}"] = tuple(label, )


            return row_labels

        def get_column_arcs(field_labels, col_id):
            col_labels = field_labels[:, col_id]
            return col_labels

        def get_block_arcs(field_labels, block_row_id, block_col_id, n):
            print(block_row_id * n, (block_row_id + 1) * n)
            block_labels = field_labels[block_row_id * n: (block_row_id + 1) * n,
                           block_col_id * n: (block_col_id + 1) * n]
            return block_labels



        arcs = np.zeros_like(self.field_array)

        for row_id in range(self.n):
            for col_id in range(self.n):
                block_row_id, block_col_id = get_block_ids(row_id, col_id, self.n)

                row_arcs = get_row_arcs(self.field_labels, row_id)
                column_arcs = get_column_arcs(self.field_labels, col_id)
                block_arcs = get_block_arcs(self.field_labels, block_row_id, block_col_id)

                running_arcs = row_arcs + column_arcs + block_arcs
                arcs[row_id][col_id] = running_arcs

        return arcs

    def prepare_to_ac(self):
        self.field_labels = self._get_initial_field_labels()
        self.arcs = self._get_arcs()

