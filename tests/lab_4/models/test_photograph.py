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

@pytest.mark.parametrize('images, masks, alpha, beta, res', [(np.array([[[[122, 123, 123],[221, 132, 221],[123, 221, 221]],
                                                                     [[11,32,34], [54,65,32], [12,36,78]],
                                                                     [[55,62,52], [1,1,1], [0,0,0]]],
                                                                     [[[123, 224, 13], [221, 212, 21], [113, 221, 121]],
                                                                     [[190, 183, 72], [255, 212, 93], [111, 61, 78]],
                                                                     [[8, 27, 52], [41, 222, 220], [200, 100, 0]]]]),
                                                        np.array([[[0,0,255],
                                                                  [0,255,0],
                                                                  [0,0,255]],
                                                                  [[0,0,0],
                                                                   [255,0,0],
                                                                   [255,0,0]]]), 1, 1,
                                                            np.array([[[[0.        , 0.        , 0.        ],
                                                                        [0.        , 0.        , 0.        ],
                                                                        [0.        , 0.        , 0.        ]],
                                                                        [[0.82745098, 2.19607843, 0.8627451 ],
                                                                        [1.43921569, 3.20784314, 0.97254902],
                                                                        [0.3254902 , 3.76470588, 2.35294118]]],

                                                                        [[[0.82745098, 2.19607843, 0.8627451 ],
                                                                        [1.43921569, 3.20784314, 0.97254902],
                                                                        [0.3254902 , 3.76470588, 2.35294118]],
                                                                        [[0.        , 0.        , 0.        ],
                                                                        [0.        , 0.        , 0.        ],
                                                                        [0.        , 0.        , 0.        ]]]]))])
def test_get_transitions_costs_true(images, masks, alpha, beta, res):
    model = PhotoGraph(images/255, masks/255, alpha, beta)
    to_compare = model._get_transitions_costs()
    np.testing.assert_array_almost_equal(to_compare, res, decimal=1)

@pytest.mark.parametrize('images, masks, alpha, beta, res', [(np.array([[[[122, 123, 123],[221, 132, 221],[123, 221, 221]],
                                                                     [[11,32,34], [54,65,32], [12,36,78]],
                                                                     [[55,62,52], [1,1,1], [0,0,0]]],
                                                                     [[[123, 224, 13], [221, 212, 21], [113, 221, 121]],
                                                                     [[190, 183, 72], [255, 212, 93], [111, 61, 78]],
                                                                     [[8, 27, 52], [41, 222, 220], [200, 100, 0]]]]),
                                                        np.array([[[0,0,255],
                                                                  [0,255,0],
                                                                  [0,0,255]],
                                                                  [[0,0,0],
                                                                   [255,0,0],
                                                                   [255,0,0]]]), 1, 1,
                                                              np.array([[[0., 1., 0.],
                                                                      [0., 1., 1.],
                                                                      [0., 1., 0.]],

                                                                     [[0., 1.8627451, 0.8627451],
                                                                      [0., 2., 1.],
                                                                      [0., 2., 1.]]])
                                                              )])
def test_get_total_costs_true(images, masks, alpha, beta, res):
    model = PhotoGraph(images / 255, masks / 255, alpha, beta)
    model.pixels_costs = model._get_pixels_costs()
    model.transitions_costs = model._get_transitions_costs()
    to_compare = model._get_total_costs()
    np.testing.assert_array_almost_equal(to_compare, res, decimal=1)