import numpy as np


def check_upper_entrances(upper_term, upper_term_shape, view):
    for i in range(upper_term_shape[0]):
        if upper_term not in view[i][0]:
            return False

    return True


def check_lower_entrances(lower_term, lower_term_shape, upper_term_shape, view):
    bias = upper_term_shape[0]

    for i in range(lower_term_shape[0]):
        if lower_term not in view[bias + i][0]:
            return False

    return True


def check_right_entrances(right_term, right_term_shape, view):
    for i in range(right_term_shape[1]):
        if right_term not in view[0][i]:
            return False

    return True


def check_left_entrances(left_term, left_term_shape, right_term_shape, view):
    bias = right_term_shape[1]

    for i in range(left_term_shape[1]):
        if left_term not in view[0][bias + i]:
            return False

    return True


def update_markup(markup, view_row_id, view_col_id, result_shape, result, replace):
    new_markup = markup.copy()
    for i in range(result_shape[0]):
        for j in range(result_shape[1]):
            if replace:
                new_markup[view_row_id + i, view_col_id + j] = list([result])
            else:
                if result not in new_markup[view_row_id + i, view_col_id + j]:
                    new_markup[view_row_id + i, view_col_id + j].append(result)

    return new_markup


def vconcat(markup, vconcat_rules_with_shapes, views):
    for view_row_id, views_row in enumerate(views):
        for view_col_id, view in enumerate(views_row):
            for rule in vconcat_rules_with_shapes:
                upper_term, lower_term = rule.get("terms")
                upper_term_shape, lower_term_shape = rule.get("terms_shapes")

                upper_term_checked = check_upper_entrances(upper_term, upper_term_shape, view)
                lower_term_checked = check_lower_entrances(lower_term, lower_term_shape, upper_term_shape, view)

                if upper_term_checked and lower_term_checked:
                    result = rule.get("result")
                    result_shape = rule.get("result_shape")
                    replace = rule.get("replace")

                    markup = update_markup(markup, view_row_id, view_col_id, result_shape, result, replace)
    return markup


def hconcat(markup, hconcat_rules_with_shapes, views):
    for view_row_id, views_row in enumerate(views):
        for view_col_id, view in enumerate(views_row):
            for rule in hconcat_rules_with_shapes:
                right_term, left_term = rule.get("terms")
                right_term_shape, left_term_shape = rule.get("terms_shapes")

                right_term_checked = check_right_entrances(right_term, right_term_shape, view)
                left_term_checked = check_left_entrances(left_term, left_term_shape, right_term_shape, view)

                if right_term_checked and left_term_checked:
                    result = rule.get("result")
                    result_shape = rule.get("result_shape")
                    replace = rule.get("replace")

                    markup = update_markup(markup, view_row_id, view_col_id, result_shape, result, replace)
    return markup


