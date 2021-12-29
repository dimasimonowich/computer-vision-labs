import pytest
import numpy as np
import cv2 as cv
from lab_5.models.grammars_core import Grammars

zero_0 = cv.imread("lab_5/data/zero_0.png")[:, :, 0]
zero_1 = cv.imread("lab_5/data/zero_1.png")[:, :, 0]
zero_2 = cv.imread("lab_5/data/zero_2.png")[:, :, 0]
one_0 = cv.imread("lab_5/data/one_0.png")[:, :, 0]
one_1 = cv.imread("lab_5/data/one_1.png")[:, :, 0]
one_2 = cv.imread("lab_5/data/one_2.png")[:, :, 0]
two_0 = cv.imread("lab_5/data/two_0.png")[:, :, 0]
incorrect = cv.imread("lab_5/data/incorrect_0.png")[:, :, 0]

objects = {
    "0": (1, 1),
    "1": (1, 1),
    "-1": (1, 1),
    "00_col": (2, 1),
    "01_col": (2, 1),
    "10_col": (2, 1),
    "11_col": (2, 1),
    "empty_col": (2, 1),
    "fine_col": (3, 1),
    "border": (3, 1),
    "needs_transition_col": (3, 1),
    "transition_applied_col": (3, 1),
    "blocked": (3, 2),
    "fine_image": (3, 2)
}

rules = [
    ("vconcat", ["0", "0"], "00_col", False),
    ("vconcat", ["0", "1"], "01_col", False),
    ("vconcat", ["1", "0"], "10_col", False),
    ("vconcat", ["1", "1"], "11_col", False),
    ("vconcat", ["-1", "-1"], "empty_col", False),

    ("vconcat", ["00_col", "0"], "fine_col", True),
    ("vconcat", ["01_col", "1"], "fine_col", True),
    ("vconcat", ["10_col", "1"], "fine_col", True),
    ("vconcat", ["empty_col", "-1"], "border", True),

    ("vconcat", ["11_col", "1"], "needs_transition_col", False),
    ("vconcat", ["11_col", "1"], "transition_applied_col", False),

    ("vconcat", ["11_col", "0"], "needs_transition_col", True),
    ("vconcat", ["10_col", "0"], "needs_transition_col", True),

    ("vconcat", ["01_col", "0"], "transition_applied_col", True),
    ("vconcat", ["00_col", "1"], "transition_applied_col", True),

    ("hconcat", ["border", "needs_transition_col"], "fine_image", False),
    ("hconcat", ["transition_applied_col", "needs_transition_col"], "fine_image", False),
    ("hconcat", ["needs_transition_col", "needs_transition_col"], "fine_image", False),

    ("hconcat", ["border", "fine_col"], "fine_image", False),
    ("hconcat", ["fine_col", "fine_col"], "fine_image", False),

    ("hconcat", ["fine_col", "needs_transition_col"], "blocked", True),
    ("hconcat", ["transition_applied_col", "fine_col"], "blocked", True),
    ("hconcat", ["transition_applied_col", "border"], "blocked", True)
]

upper_term = cv.hconcat([one_0, one_0, one_0, one_0, one_1, one_0])
lower_term = cv.hconcat([zero_0, one_0, zero_0, one_0, zero_0, zero_1])
result = cv.hconcat([zero_2, zero_1, one_0, zero_1, one_0, one_0])

ones = np.array([one_0, one_1, one_2])/255
zeros = np.array([zero_0, zero_1, zero_2])/255

train_data = [(zeros, "0"), (ones, "1")]

@pytest.mark.parametrize('image', [two_0/255, incorrect/255])
def test_classify_true(image):
    model = Grammars(train_data, rules, objects)
    to_compare = model._classify(image)
    assert to_compare == None

