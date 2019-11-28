from XPLAIN_utils.LACE_utils.LACE_utils2 import *
from XPLAIN_utils.LACE_utils.LACE_utils4 import *
from XPLAIN_utils.LACE_utils.LACE_utils5 import *


def get_rule_index(rule, instance):
    rule_index = []
    for r in rule:
        attribute_index = get_attribute_index(r, instance)
        if attribute_index == -1:
            return False
        rule_index.append(attribute_index)
    return rule_index


def get_attribute_index(attribute, instanceI):
    index = 0
    for i in instanceI.domain.attributes:
        if i.name == attribute:
            return index
        index += 1
    return -1


class XPLAIN_explanation:
    def __init__(self, explainer, target_class, instance, diff_single, impo_rules,
                 instance_id, k, error, difference_map, impo_rules_complete):
        self.XPLAIN_explainer_o = explainer
        self.instance_id = instance_id
        self.diff_single = diff_single
        self.impo_rules = impo_rules
        self.map_difference = deepcopy(difference_map)
        self.k = k
        self.error = error
        self.impo_rules_complete = deepcopy(impo_rules_complete)
        self.instance = instance
        self.target_class = target_class
        self.instance_class_index = self.XPLAIN_explainer_o.get_class_index(self.target_class)

        c1 = self.XPLAIN_explainer_o.classifier(self.instance, True)[0]
        self.prob = c1[self.instance_class_index]

        self.pred_str = str(round(self.prob, 2))

