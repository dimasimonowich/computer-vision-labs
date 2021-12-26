from .grammars_core import Grammars
from .rules import hconcat, vconcat
import numpy as np


class BinSumCheck(Grammars):
    def __init__(self, train_data, rules, objects, max_iter=100):
        super().__init__(train_data, rules, objects)
        self.max_iter = max_iter

    def kek_predict(self, image):
        initial_markup = self.get_initial_markup(image)

        term_0 = int(f"{int(initial_markup[0][0])}{int(initial_markup[0][1])}{int(initial_markup[0][2])}", base=2)
        term_1 = int(f"{int(initial_markup[1][0])}{int(initial_markup[1][1])}{int(initial_markup[1][2])}", base=2)
        result = int(f"{int(initial_markup[2][0])}{int(initial_markup[2][1])}{int(initial_markup[2][2])}", base=2)

        int_sum = term_0 + term_1

        return result == int_sum

    def _parse_rules(self):
        vconcat_rules = []
        hconcat_rules = []
        result_shapes = []

        for rule in self.rules:
            terms = rule[1]
            result = rule[2]
            replace = rule[3]
            terms_shapes = [self.objects.get(terms[0]), self.objects.get(terms[1])]
            result_shape = self.objects.get(result)

            parsed_rule = {
                "terms": terms,
                "result": result,
                "replace": replace,
                "terms_shapes": terms_shapes,
                "result_shape": result_shape
            }

            result_shapes.append(result_shape)

            if rule[0] == "vconcat":
                vconcat_rules.append(parsed_rule)
            else:
                hconcat_rules.append(parsed_rule)

        return vconcat_rules, hconcat_rules, np.unique(result_shapes, axis=0)

    def predict(self, image):
        def get_image_views(image, shapes):
            return np.lib.stride_tricks.sliding_window_view(image, shapes)

        markup = self.get_initial_markup(image)
        vconcat_rules, hconcat_rules, unique_shapes = self._parse_rules()

        for shapes in unique_shapes:
            if -1 not in shapes:
                views = get_image_views(markup, shapes)
                vconcat_rules_with_shapes = []
                hconcat_rules_with_shapes = []

                for vconcat_rule in vconcat_rules:
                    if np.all(vconcat_rule.get("result_shape") == shapes):
                        vconcat_rules_with_shapes.append(vconcat_rule)

                for hconcat_rule in hconcat_rules:
                    if np.all(hconcat_rule.get("result_shape") == shapes):
                        hconcat_rules_with_shapes.append(hconcat_rule)

                markup = vconcat(markup, vconcat_rules_with_shapes, views)
                markup = hconcat(markup, hconcat_rules_with_shapes, views)

