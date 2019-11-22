from snapshottest import TestCase
from api import get_explanation


class TestGet_explanation(TestCase):
    def test_get_explanation(self):
        explanation = get_explanation()
        self.assertMatchSnapshot(
            (explanation.n_inst,
             explanation.diff_single,
             explanation.impo_rules,
             explanation.impo_rulesUser,
             explanation.map_difference,
             explanation.map_differenceUser,
             explanation.KNN,
             explanation.error,
             explanation.impo_rules_complete,
             explanation.impo_rules_completeUser,
             explanation.instance,
             explanation.errorUser,
             explanation.targetClass,
             explanation.indexI,
             explanation.pred,
             explanation.pred_str))

