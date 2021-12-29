import cv2 as cv
import pytest
import numpy as np
from lab_5.models.bin_sum_check import BinSumCheck


mini_0 = cv.imread("lab_5/data/mini_0.png")[:, :, 0]
mini_1 = cv.imread("lab_5/data/mini_1.png")[:, :, 0]


mini_upper_term = cv.hconcat([mini_0, mini_1])
mini_lower_term = cv.hconcat([mini_1, mini_1])
mini_result = cv.hconcat([mini_1, mini_0])

mini_addition = cv.vconcat([mini_upper_term, mini_lower_term, mini_result])/255

mini_ones = np.array([mini_1])/255
mini_zeros = np.array([mini_0])/255

mini_train_data = [(mini_zeros, "0"), (mini_ones, "1")]

mini_objects = {
    "0": (1, 1),
    "00_col": (2, 1),
    "fine_col": (3, 1),
    "fine_image": (3, 2)
}

mini_rules = [
    ("vconcat", ["0", "0"], "00_col", False),
    ("vconcat", ["00_col", "0"], "fine_col", True),
    ("hconcat", ["fine_col", "fine_col"], "fine_image", False)]


@pytest.mark.parametrize('rules, res', [(mini_rules, ([{'terms': ['0', '0'],
   'result': '00_col',
   'replace': False,
   'terms_shapes': [(1, 1), (1, 1)],
   'result_shape': (2, 1)},
  {'terms': ['00_col', '0'],
   'result': 'fine_col',
   'replace': True,
   'terms_shapes': [(2, 1), (1, 1)],
   'result_shape': (3, 1)}],
 [{'terms': ['fine_col', 'fine_col'],
   'result': 'fine_image',
   'replace': False,
   'terms_shapes': [(3, 1), (3, 1)],
   'result_shape': (3, 2)}],
 np.array([[2, 1],
        [3, 1],
        [3, 2]])))])
def test_parse_rules_true(rules, res):
    model = BinSumCheck(mini_train_data, mini_rules, mini_objects,
                  final_symbol="fine_image",
                  transition_symbol="needs_transition_col",
                  early_stopping_symbol="blocked")
    to_compare = model._parse_rules()
    np.equal(to_compare, res, dtype=tuple)


@pytest.mark.parametrize('mini_rm, mini_tr, test_markup', [(
                                                            np.array([[-1, -1],
                                                                      [-1, -1],
                                                                      [-1, -1]]),
                                                                     False,
                                                            np.array([['-1', '0', '1', '-1'],
                                                                      ['-1', '1', '1', '-1'],
                                                                      ['-1', '1', '0', '-1']]))])
def test_predict_true(mini_rm, mini_tr, test_markup):
    model = BinSumCheck(mini_train_data, mini_rules, mini_objects,
                        final_symbol="fine_image",
                        transition_symbol="needs_transition_col",
                        early_stopping_symbol="blocked")
    mini_res_markup, mini_req_trans, mini_markup = model.predict(mini_addition)

    assert mini_markup.shape == test_markup.shape

    for row_id in range(mini_markup.shape[0]):
        for col_id in range(mini_markup.shape[1]):
            assert mini_markup[row_id][col_id][0] == test_markup[row_id][col_id]

    assert mini_tr == mini_req_trans

    np.testing.assert_array_almost_equal( mini_res_markup, mini_rm, decimal=1)



