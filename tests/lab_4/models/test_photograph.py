import pytest
import numpy as np
from lab_4.models.photograph import PhotoGraph

@pytest.mark.parametrize('image, mask, alpha, beta', [([[[123, 123, 123],[221, 132, 221],[123, 221, 221]],
                                                         [[12,32,34], [54,65,32], [12,36,78]],
                                                         [[56,62,52], [1,1,1], [0,0,0]]],
                                                        [[0,0,255],
                                                         [0,255,0],
                                                         [0,0,255]], 1, 1)])
def test_init_true(image, mask, alpha, beta):
    with pytest.raises(AssertionError):
        model = PhotoGraph(image, mask, alpha, beta)

@pytest.mark.parametrize('image, mask, alpha, beta', [( [56, 2, -1, -1])])
def test_init_false(image, mask, alpha, beta):
    with pytest.raises(AssertionError):
        model = PhotoGraph(image, mask, alpha, beta)