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

@pytest.mark.parametrize('image, mask, alpha, beta, res', [(np.array([[[122, 123, 123],[221, 132, 221],[123, 221, 221]],
                                                                     [[11,32,34], [54,65,32], [12,36,78]],
                                                                     [[55,62,52], [1,1,1], [0,0,0]]]),
                                                        np.array([[0,0,255],
                                                                  [0,255,0],
                                                                  [0,0,255]]), 1, 1,
                                                            np.array([[[1., 1., 0.],
                                                                     [1., 0., 1.],
                                                                     [1., 1., 0.]]]))])
def test_get_pixels_costs_true(image, mask, alpha, beta, res):
    model = PhotoGraph(np.array([image])/255, np.array([mask])/255, alpha, beta)
    to_compare = model._get_pixels_costs()
    np.testing.assert_array_almost_equal(to_compare, res, decimal=1)