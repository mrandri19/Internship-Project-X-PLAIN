from XPLAIN_utils.LACE_utils.LACE_utils2 import *
from XPLAIN_utils.LACE_utils.LACE_utils4 import *
from XPLAIN_utils.LACE_utils.LACE_utils5 import *


def getIndexRule(rule, instance):
    rule_Index = []
    for r in rule:
        indexA = getIndexAttribute(r, instance)
        if indexA == -1:
            return False
        rule_Index.append(indexA)
    return rule_Index


def getIndexAttribute(attribute, instanceI):
    index = 1
    for i in instanceI.domain.attributes:
        if i.name == attribute:
            return index
        index = index + 1
    return -1


class XPLAIN_explanation:
    def __init__(self, XPLAIN_explainer_o, targetClass, instance, diff_single, impo_rules, n_inst, KNN, error,
                 map_difference, impo_rules_complete):

        self.XPLAIN_explainer_o = XPLAIN_explainer_o
        self.n_inst = n_inst
        self.diff_single = diff_single
        self.impo_rules = impo_rules
        self.impo_rulesUser = deepcopy(impo_rules)
        self.map_difference = deepcopy(map_difference)
        self.map_differenceUser = deepcopy(map_difference)
        self.KNN = KNN
        self.error = error
        self.impo_rules_complete = deepcopy(impo_rules_complete)
        self.impo_rules_completeUser = deepcopy(impo_rules_complete)
        self.instance = instance
        self.errorUser = error
        c1 = self.XPLAIN_explainer_o.classifier(self.instance, True)[0]
        self.targetClass = targetClass
        self.indexI = self.XPLAIN_explainer_o.get_class_index(self.targetClass)
        self.pred = c1[self.indexI]
        self.pred_str = str(round(c1[self.indexI], 2))

    def getNewExplanation(self, rule, update=False):
        rule_Index = getIndexRule(rule, self.instance)
        rule_print = "{"
        for r in rule:
            rule_print = rule_print + r + "=" + self.instance[r].value + ", "
        rule_print = rule_print[:-2] + "}"

        print("Rule inserted:", rule_print)

        if update:
            if rule_Index in self.impo_rules_completeUser:
                print("Rule already present")
                # Solo imporules+user
                plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,
                               self.targetClass, self.XPLAIN_explainer_o.classif, self.map_differenceUser,
                               self.pred_str, self.impo_rules)

            else:
                user_rules = self.getUserRules(rule_Index, update)

                map_difference_new = {}
                map_difference_new = compute_prediction_difference_subset_only_existing(
                    self.XPLAIN_explainer_o.training_dataset, self.instance, user_rules,
                    self.XPLAIN_explainer_o.classifier, self.indexI, map_difference_new)

                self.updateUserRules(user_rules, map_difference_new, rule_Index)

                error, PI_rel2 = computeApproxErrorRule(self.XPLAIN_explainer_o.mappa_class, self.pred,
                                                        self.diff_single, user_rules, self.targetClass,
                                                        self.map_differenceUser)
                # print("error", error)
                plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,
                               self.targetClass, self.XPLAIN_explainer_o.classif, self.map_differenceUser,
                               self.pred_str, self.impo_rules, user_rules)

                return rule_Index

        else:
            if rule_Index in self.impo_rules_complete:
                print("Rule already present")

                plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,
                               self.targetClass, self.XPLAIN_explainer_o.classif, self.map_difference, self.pred_str,
                               self.impo_rules)

            else:
                user_rules = self.getUserRules(rule_Index, update)

                map_difference_new = {}
                map_difference_new = compute_prediction_difference_subset_only_existing(
                    self.XPLAIN_explainer_o.training_dataset, self.instance, user_rules,
                    self.XPLAIN_explainer_o.classifier, self.indexI, map_difference_new)

                tmp = deepcopy(self.map_difference)
                map_difference_new.update(tmp)

                error, PI_rel2 = computeApproxErrorRule(self.XPLAIN_explainer_o.mappa_class, self.pred,
                                                        self.diff_single, user_rules, self.targetClass,
                                                        map_difference_new)
                # print("error",error)
                # print(map_difference_new)
                plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,
                               self.targetClass, self.XPLAIN_explainer_o.classif, map_difference_new, self.pred_str,
                               self.impo_rules, user_rules)

                return rule_Index
