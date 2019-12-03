from snapshottest import TestCase

from api import get_explanation


class TestGet_explanation(TestCase):
    def test_get_explanation_zoo_random_forest(self):
        e = get_explanation("zoo", "rf")
        self.assertMatchSnapshot((
            e.XPLAIN_explainer_o,
            e.instance_id,
            e.diff_single,
            e.map_difference,
            e.k,
            e.error,
            e.instance,
            e.target_class,
            e.instance_class_index,
            e.prob
        ))

    def test_get_explanation_zoo_naive_bayes(self):
        e = get_explanation("zoo", "nb")
        self.assertMatchSnapshot((
            e.XPLAIN_explainer_o,
            e.instance_id,
            e.diff_single,
            e.map_difference,
            e.k,
            e.error,
            e.instance,
            e.target_class,
            e.instance_class_index,
            e.prob
        ))

    def test_get_explanation_adult_naive_bayes(self):
        e = get_explanation("datasets/adult_d.arff", "nb")
        self.assertMatchSnapshot((
            e.XPLAIN_explainer_o,
            e.instance_id,
            e.diff_single,
            e.map_difference,
            e.k,
            e.error,
            e.instance,
            e.target_class,
            e.instance_class_index,
            e.prob
        ))
