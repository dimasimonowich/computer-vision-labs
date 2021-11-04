import pytest
import numpy as np
from lab_2.models.koz_model import KozModel
model = KozModel()
@pytest.mark.parametrize("vec", [[0, 2, 4, 4],
                                 [-0.1, 2, 4],
                                 [22, 4, 6]])
def test_init_with_first_vec_true(vec):
    model.init_with_first_vec(vec)
    assert model.beta == vec[0]


@pytest.mark.parametrize("vec", [[]])
def test_init_with_first_vec_false(vec):
    with pytest.raises(AssertionError):
        model.init_with_first_vec(vec)

@pytest.mark.parametrize("vec, ans", [([[0, 2], [0.1, 0.2]],
                                       [[1, 0, 2, 0, 0, 0, 4], [1, 0.1, 0.2, 0.01, 0.02, 0.02, 0.04]])])

def test_load_points_to_vec_true(vec, ans):
    for i in range(len(vec)):
        np.testing.assert_array_almost_equal(model.load_points_to_vec(vec)[i], ans[i], decimal= 2)


@pytest.mark.parametrize("initial_beta, ans", [(np.array([1, 1, 1, 1, 1, 1, 1]),
                                                np.array([[0., 0., 0., 0.5, -0.5, -0.5, 0.5],
                                                       [0., 0., 0., 0.5, 0.5, 0.5, 0.5]])),
                                               (np.array([1, 0, 0, 0, 0, 0, 0]),
                                                np.array([[0., 0., 0., 1., 0., 0., 0.],
                                                       [0., 0., 0., 0., 0., 0., 1.]]))])

def test_load_points_to_vec_true(initial_beta, ans):
    test_classifier = KozModel()
    test_classifier.beta = initial_beta
    test_ans = test_classifier.create_dummy_vecs()
    np.testing.assert_array_almost_equal(test_ans, ans, decimal= 2)