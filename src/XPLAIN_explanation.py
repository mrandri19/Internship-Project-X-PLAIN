from copy import deepcopy

MT = 1


class XPLAIN_explanation:
    def __init__(self, explainer, target_class, instance, diff_single, k, error, difference_map):
        self.XPLAIN_explainer_o = explainer
        self.diff_single = diff_single
        self.map_difference = deepcopy(difference_map)
        self.k = k
        self.error = error
        self.instance = instance
        self.target_class = target_class
        self.instance_class_index = explainer.get_class_index(self.target_class)

        self.prob = \
            self.XPLAIN_explainer_o.classifier[MT].predict_proba(instance.x.reshape(1, -1))[0][
                self.instance_class_index]
