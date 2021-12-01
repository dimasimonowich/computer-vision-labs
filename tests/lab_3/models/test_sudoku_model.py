import pytest
import numpy as np
from lab_3.models.sudoku import Sudoku

@pytest.mark.parametrize("field",  [[[1,0,2,3],
                                    [2,3,0,2],
                                    [3,2,3,1]],

                                    [[2,0],
                                     [0,0]],

                                    [[6,6,6,6],
                                    [0,0,0,0],
                                    [10,1,2,10],
                                    [5,2,6,7,]],

                                    [[-2,-3,5,0],
                                    [-2,-6,-7,0],
                                    [-1,-2,0,0],
                                    [0,0,-1,2]]])
def test_init_false(field):
    with pytest.raises(AssertionError):
        model = Sudoku(field)


