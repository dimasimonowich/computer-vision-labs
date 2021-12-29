DEFAULT_FINAL_SYMBOL = "fine_image"
DEFAULT_TRANSITION_SYMBOL = "needs_transition_col"
DEFAULT_EARLY_STOPPING_SYMBOL = "blocked"

DEFAULT_OBJECTS = {
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

DEFAULT_RULES = [
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

