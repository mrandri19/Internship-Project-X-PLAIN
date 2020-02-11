from snapshottest import TestCase

from src.XPLAIN_explainer import XPLAIN_explainer


def get_explanation(dataset, classifier):
    explainer = XPLAIN_explainer(dataset, classifier, random_explain_dataset=True)
    instance = explainer.explain_dataset[0]
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
        e = get_explanation("datasets/adult_d.arff", "nb")
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
