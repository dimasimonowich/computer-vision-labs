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


def vertical_update_markup(markup, view_row_id, view_col_id, result_shape, result, replace):
    new_markup = markup.copy()
    for i in range(result_shape[0]):
        for j in range(result_shape[1]):
            if replace:
                new_markup[view_row_id + i, view_col_id + j] = list([result])
            else:
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
                    print(f"\nview: {view}")
                    print(f"rule: {rule}")
                    print(view_row_id, view_col_id)
                    print(f"Old markup: {markup}")

                    result = rule.get("result")
                    result_shape = rule.get("result_shape")
                    replace = rule.get("replace")

                    markup = vertical_update_markup(markup, view_row_id, view_col_id, result_shape, result, replace)
                    print(f"New markup: {markup}")

    return markup


def hconcat(markup, hconcat_rules_with_shapes, views):
    for view_row_id, views_row in enumerate(views):
        for view_col_id, view in enumerate(views_row):
            for rule in hconcat_rules_with_shapes:
                upper_term, lower_term = rule.get("terms")
                upper_term_shape, lower_term_shape = rule.get("terms_shapes")

                upper_term_checked = check_upper_entrances(upper_term, upper_term_shape, view)
                lower_term_checked = check_lower_entrances(lower_term, lower_term_shape, upper_term_shape, view)

                if upper_term_checked and lower_term_checked:
                    print(f"\nview: {view}")
                    print(f"rule: {rule}")
                    print(view_row_id, view_col_id)
                    print(f"Old markup: {markup}")

                    result = rule.get("result")
                    result_shape = rule.get("result_shape")
                    replace = rule.get("replace")

                    markup = vertical_update_markup(markup, view_row_id, view_col_id, result_shape, result, replace)
                    print(f"New markup: {markup}")
    return markup


