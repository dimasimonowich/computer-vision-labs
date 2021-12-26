from .grammars_core import Grammars


class BinSumCheck(Grammars):
    def __init__(self, train_data, rules):
        super().__init__(train_data, rules)

    def kek_predict(self, image):
        initial_markup = self.get_initial_markup(image)

        term_0 = int(f"{int(initial_markup[0][0])}{int(initial_markup[0][1])}{int(initial_markup[0][2])}", base=2)
        term_1 = int(f"{int(initial_markup[1][0])}{int(initial_markup[1][1])}{int(initial_markup[1][2])}", base=2)
        result = int(f"{int(initial_markup[2][0])}{int(initial_markup[2][1])}{int(initial_markup[2][2])}", base=2)

        int_sum = term_0 + term_1

        return result == int_sum
