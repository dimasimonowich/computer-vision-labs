import pytest
import numpy as np
import cv2 as cv
from lab_5.models.rules import vconcat
from lab_5.models.rules import hconcat
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

test_markup = np.empty((3,8), dtype=object)

for i in range(test_markup.shape[0]):
    for j in range(test_markup.shape[1]):
        test_markup[i, j] = list(["-1"])



@pytest.mark.parametrize('markup, vconcat_rules_with_shapes, views, res', [(test_markup,
[{'terms': ['0', '0'], 'result': '00_col', 'replace': False, 'terms_shapes': [(1, 1), (1, 1)], 'result_shape': (2, 1)}, {'terms': ['0', '1'], 'result': '01_col', 'replace': False, 'terms_shapes': [(1, 1), (1, 1)], 'result_shape': (2, 1)}, {'terms': ['1', '0'], 'result': '10_col', 'replace': False, 'terms_shapes': [(1, 1), (1, 1)], 'result_shape': (2, 1)}, {'terms': ['1', '1'], 'result': '11_col', 'replace': False, 'terms_shapes': [(1, 1), (1, 1)], 'result_shape': (2, 1)}, {'terms': ['-1', '-1'], 'result': 'empty_col', 'replace': False, 'terms_shapes': [(1, 1), (1, 1)], 'result_shape': (2, 1)}],

                                                                            [[[[list(['-1'])],
                                                                               [list(['-1'])]],
                                                                                [[list(['0'])],
                                                                                [list(['1'])]],
                                                                               [[list(['1'])],
                                                                                [list(['1'])]],
                                                                               [[list(['-1'])],
                                                                                [list(['-1'])]]],
                                                                               [[[list(['-1'])],
                                                                                 [list(['-1'])]],
                                                                                 [[list(['1'])],
                                                                                 [list(['1'])]],
                                                                                 [[list(['1'])],
                                                                                 [list(['0'])]],
                                                                                 [[list(['-1'])],
                                                                                 [list(['-1'])]]]],
                                                                        np.array([['-1', '-1', '-1', '-1', '-1',
                                                                                  '-1', '-1', '-1'],
                                                                                  ['-1', '-1', '-1', '-1', '-1',
                                                                                  '-1', '-1', '-1'],
                                                                                  ['-1', '-1', '-1', '-1', '-1',
                                                                                  '-1', '-1', '-1']]))])
def test_vconcut_true(markup, vconcat_rules_with_shapes, views, res):
    model = BinSumCheck(mini_train_data, mini_rules, mini_objects,
                        final_symbol="fine_image",
                        transition_symbol="needs_transition_col",
                        early_stopping_symbol="blocked")
    to_compare = vconcat(markup, vconcat_rules_with_shapes, views)
    assert res.shape == to_compare.shape

    for row_id in range(to_compare.shape[0]):
        for col_id in range(to_compare.shape[1]):
            assert to_compare[row_id][col_id][0] == res[row_id][col_id]


test_markup2 = np.empty((3,4), dtype=object)

for i in range(test_markup2.shape[0]):
    for j in range(test_markup2.shape[1]):
        test_markup2[i, j] = list(["-1"])


@pytest.mark.parametrize('markup, hconcat_rules_with_shapes, views, res', [(test_markup2,
[{'terms': ['fine_col', 'fine_col'], 'result': 'fine_image', 'replace': False, 'terms_shapes': [(3, 1), (3, 1)], 'result_shape': (3, 2)}],
                                                                        [[[[list(['-1']), list(['0'])],
                                                                                [list(['-1']), list(['1'])],
                                                                                [list(['-1']), list(['1'])]],
                                                                                [[list(['0']), list(['1'])],
                                                                                [list(['1']), list(['1'])],
                                                                                [list(['1']),list(['0'])]],
                                                                                [[list(['1']), list(['-1'])],
                                                                                [list(['1']), list(['-1'])],
                                                                                [list(['0']), list(['-1'])]]]],
                                                                            np.array([['-1', '-1', '-1', '-1'],
                                                                             ['-1', '-1', '-1', '-1'],
                                                                             ['-1', '-1', '-1', '-1']]))])
def test_hconcut_true(markup, hconcat_rules_with_shapes, views, res):
    model = BinSumCheck(mini_train_data, mini_rules, mini_objects,
                        final_symbol="fine_image",
                        transition_symbol="needs_transition_col",
                        early_stopping_symbol="blocked")
    to_compare = hconcat(markup, hconcat_rules_with_shapes, views)
    assert res.shape == to_compare.shape

    for row_id in range(to_compare.shape[0]):
        for col_id in range(to_compare.shape[1]):
            assert to_compare[row_id][col_id][0] == res[row_id][col_id]
