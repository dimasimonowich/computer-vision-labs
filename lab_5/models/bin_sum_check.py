from .grammars_core import Grammars
from .rules import hconcat, vconcat
import numpy as np
from matplotlib import pyplot as plt


class BinSumCheck(Grammars):
    def __init__(self, train_data, rules, objects, final_symbol, transition_symbol, early_stoping_symbol=None):
        super().__init__(train_data, rules, objects)
        self.final_symbol = final_symbol
        self.early_stoping_symbol = early_stoping_symbol
        self.transition_symbol = transition_symbol

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

    def _check_result(self, markup):
        resulting_markup = np.ones(markup.shape, dtype=int) * (-1)
        requires_transition = False

        for row_id, labels_row in enumerate(markup):
            for col_id, label in enumerate(labels_row):

                if self.final_symbol in label:
                    resulting_markup[row_id][col_id] = 1

                if self.early_stoping_symbol:
                    if self.early_stoping_symbol in label:
                        resulting_markup[row_id][col_id] = 0

                if col_id == 1 and self.transition_symbol in label:
                    requires_transition = True

        return resulting_markup[:, 1:-1], requires_transition

    def predict(self, image):
        def get_image_views(image, shapes):
            return np.lib.stride_tricks.sliding_window_view(image, shapes)

        def add_borders(markup):
            empty_col = np.empty((markup.shape[0], 1), dtype=object)

            for i in range(markup.shape[0]):
                empty_col[i, 0] = list(["-1"])

            return np.hstack([empty_col, markup, empty_col])

        markup = self.get_initial_markup(image)
        markup = add_borders(markup)

        vconcat_rules, hconcat_rules, unique_shapes = self._parse_rules()

        for shapes in unique_shapes:
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

        resulting_markup, requires_transition = self._check_result(markup)

        return resulting_markup, requires_transition, markup

    @staticmethod
    def draw_results(image, resulting_markup, requires_transition):
        addition_correct = True
        colored_image = np.stack([image, image, image], axis=2)

        if np.any(resulting_markup == 0):
            addition_correct = False

            shapes_ratio = image.shape[1] / resulting_markup.shape[1]
            for idx, element in enumerate(resulting_markup[0]):
                if element == 0:
                    colored_image[:, int(idx * shapes_ratio):int((idx + 1) * shapes_ratio), 0] = \
                        np.where(colored_image[:, int(idx * shapes_ratio):int((idx + 1) * shapes_ratio), 0] == 0, 1,
                                 colored_image[:, int(idx * shapes_ratio):int((idx + 1) * shapes_ratio), 0])

        plt.imshow(colored_image)
        plt.show()

        if addition_correct:
            print("Addition is correct!")
            if requires_transition:
                print("Transition required!")
            else:
                print("No transition!")
        else:
            print("Addition is incorrect!")

