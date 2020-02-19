from os.path import join

from snapshottest import TestCase

from src import DEFAULT_DIR
from src.XPLAIN_explainer import XPLAIN_explainer, OT
from src.XPLAIN_explanation import XPLAIN_explanation


def get_explanation(dataset_name: str, classifier_name: str) -> XPLAIN_explanation:
    explainer = XPLAIN_explainer(dataset_name, classifier_name, random_explain_dataset=True)
    instance = explainer.explain_dataset[OT][0]
    return explainer.explain_instance(instance, target_class=instance.get_class().value)


class TestGet_explanation(TestCase):
    def test_get_explanation_zoo_random_forest(self):
        e = get_explanation("zoo", "rf")
        self.assertMatchSnapshot((
            e.XPLAIN_explainer_o,
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
        e = get_explanation(join(DEFAULT_DIR, "datasets/adult_d.arff"), "nb")
        self.assertMatchSnapshot((
            e.XPLAIN_explainer_o,
            e.diff_single,
            e.map_difference,
            e.k,
            e.error,
            e.instance,
            e.target_class,
            e.instance_class_index,
            e.prob
        ))
