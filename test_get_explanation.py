from snapshottest import TestCase

from api import get_explanation


class TestGet_explanation(TestCase):
    def test_get_explanation_zoo_random_forest(self):
        e = get_explanation("zoo", "rf")
        self.assertMatchSnapshot(
            (e.n_inst,
             e.diff_single,
             e.impo_rules,
             e.impo_rulesUser,
             e.map_difference,
             e.map_differenceUser,
             e.KNN,
             e.error,
             e.impo_rules_complete,
             e.impo_rules_completeUser,
             e.instance,
             e.errorUser,
             e.targetClass,
             e.indexI,
             e.pred,
             e.pred_str))

    def test_get_explanation_zoo_naive_bayes(self):
        e = get_explanation("zoo", "nb")
        self.assertMatchSnapshot(
            (e.n_inst,
             e.diff_single,
             e.impo_rules,
             e.impo_rulesUser,
             e.map_difference,
             e.map_differenceUser,
             e.KNN,
             e.error,
             e.impo_rules_complete,
             e.impo_rules_completeUser,
             e.instance,
             e.errorUser,
             e.targetClass,
             e.indexI,
             e.pred,
             e.pred_str))
