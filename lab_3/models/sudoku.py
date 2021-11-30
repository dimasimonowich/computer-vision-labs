import numpy as np


class SudokuBox:
    def __init__(self, row_id, col_id, field):
        self.row_id = row_id
        self.col_id = col_id

        self.labels = self._get_labels(field)
        self.related = self._get_related(field)

    def _get_labels(self, field):
        relative_row = field[self.row_id][self.col_id]

        return np.array(related)


    def _get_related(self, field):
        relative_row = field[self.row_id]
        relative_col = field[:, self.col_id]
        relative_block
        related = set(relative_row + relative_col + relative_block)

        return np.array(related)


class Sudoku:
    def __init__(self, field_array):
        assert field_array.shape[0] == field_array.shape[1]
        assert field_array.shape[0] >= 4
        assert field_array.shape[0] % 2 == 0

        self.field_array = field_array
        self.n = field_array.shape[0]

        self.field = None

    @staticmethod
    def _create_field(field):
        field = []

        for row_id in range(self.n):
            for col_id in range(self.n):
                sudoku_box = SudokuBox(row_id, col_id, field)





    # def get_relatives(self, row_id, col_id):
    #     relative_row = self.field[row_id]
    #     relative_col = self.field[:, col_id]
    #     relative_block =
    #
    #     relatives =

