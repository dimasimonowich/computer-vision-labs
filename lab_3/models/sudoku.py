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

        def get_arcs_in_fields_list(fields_list):
            arcs = []

            for i, from_field in enumerate(fields_list):
                for j, to_field in enumerate(fields_list):
                    if j > i:
                        arc = {}
                        arc['from_id'] = (from_field['row_id'], from_field['col_id'])
                        arc['to_id'] = (to_field['row_id'], to_field['col_id'])
                        arcs.append(arc)

            return arcs

        def get_block_fields(fields, block_row_id, block_col_id):
            fields_list = []

            for row in fields:
                for field in row:
                    if field.get('block_row_id') == block_row_id \
                      and field.get('block_col_id') == block_col_id:
                        fields_list.append(field)

            return fields_list

        arcs = np.zeros_like(self.field_array)

        for row_id in range(self.n):
            for col_id in range(self.n):
                row_fields = self.fields[row_id]
                col_fields = self.fields[:, col_id]
                block_fields = get_block_fields(self.fields, block_row_id, block_col_id)

                row_arcs = get_arcs_in_fields_list(row_fields)
                column_arcs = get_arcs_in_fields_list(col_fields)
                block_arcs = get_arcs_in_fields_list(block_fields)

                running_arcs = row_arcs + column_arcs + block_arcs
                arcs[row_id][col_id] = running_arcs

        return arcs

    def prepare_to_ac(self):
        self.field_labels = self._get_initial_field_labels()
        self.arcs = self._get_arcs()

