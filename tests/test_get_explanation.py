import random
from os.path import join
from typing import Tuple, List

import arff
from snapshottest import TestCase

from src import DEFAULT_DIR
from src.XPLAIN_explainer import XPLAIN_explainer
from src.dataset import Dataset


def load_arff(f) -> Dataset:
    a = arff.load(f)
    dataset = Dataset(a['data'], a['attributes'])

    return dataset


def import_dataset_arff(f, explain_indices: List[int],
                        random_explain_dataset: bool) -> Tuple[Dataset, Dataset, List[str]]:
    dataset = load_arff(f)

    dataset_len = len(dataset)
    train_indices = list(range(dataset_len))

    if random_explain_dataset:
        random.seed(1)
        # small dataset
        MAX_SAMPLE_COUNT = 100
        if dataset_len < (2 * MAX_SAMPLE_COUNT):
            samples = int(0.2 * dataset_len)
        else:
            samples = MAX_SAMPLE_COUNT

        # Randomly pick some instances to remove from the training dataset and use in the
        # explain dataset
        explain_indices = list(random.sample(train_indices, samples))
    for i in explain_indices:
        train_indices.remove(i)

    train_dataset = Dataset.from_indices(train_indices, dataset)
    explain_dataset = Dataset.from_indices(explain_indices, dataset)

    return train_dataset, explain_dataset, [str(i) for i in explain_indices]


def import_datasets_arff(f, f_explain, explain_indices: List[int],
                         random_explain_dataset: bool) -> Tuple[Dataset, Dataset, List[str]]:
    train_dataset = load_arff(f)
    explain_dataset = load_arff(f_explain)

    len_explain_dataset = len(explain_dataset)

    if random_explain_dataset:
        random.seed(7)
        explain_indices = list(random.sample(range(len_explain_dataset), 300))
        explain_dataset = Dataset.from_indices(explain_indices, explain_dataset)

    return train_dataset, explain_dataset, [str(i) for i in explain_indices]


def get_classifier(classifier_name: str):
    if classifier_name == "sklearn_nb":
        from sklearn.naive_bayes import MultinomialNB

        skl_clf = MultinomialNB()

        return skl_clf

    elif classifier_name == "sklearn_rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        pipe = make_pipeline(OneHotEncoder(), RandomForestClassifier(random_state=42))
        skl_clf = pipe

        return skl_clf

    elif classifier_name == "nn_label_enc":
        from sklearn.neural_network import MLPClassifier

        skl_clf = MLPClassifier(random_state=42, max_iter=1000)

        return skl_clf

    elif classifier_name == "nn_onehot_enc":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        pipe = make_pipeline(OneHotEncoder(), MLPClassifier(random_state=42, max_iter=1000))
        skl_clf = pipe

        return skl_clf

    else:
        raise ValueError("Classifier not available")


def get_explanation(dataset_name: str, classifier_name: str):
    explain_dataset_indices = []
    if dataset_name in [join(DEFAULT_DIR, "datasets/adult_d.arff"),
                        join(DEFAULT_DIR, "datasets/compas-scores-two-years_d.arff")]:
        with open(dataset_name) as f, open(dataset_name[:-5] + "_explain.arff") as f_explain:
            train_dataset, explain_dataset, explain_indices = import_datasets_arff(f, f_explain,
                                                                                   explain_dataset_indices,
                                                                                   True)
    else:
        with open(dataset_name) as f:
            train_dataset, explain_dataset, explain_indices = import_dataset_arff(
                f, explain_dataset_indices, True)

    clf = get_classifier(classifier_name).fit(train_dataset.X_numpy(),
                                              train_dataset.Y_numpy())
    explainer = XPLAIN_explainer(clf, train_dataset, explain_dataset)

    instance = explainer.explain_dataset.get_decoded(0)

    cc = explainer.explain_dataset.class_column_name()
    target_class = instance[cc]

    return explainer.explain_instance(instance, target_class=target_class)


class TestGet_explanation(TestCase):
    def test_get_explanation_zoo_random_forest(self):
        e = get_explanation(join(DEFAULT_DIR, "datasets/zoo.arff"), "sklearn_rf")
        self.assertMatchSnapshot((
            e['XPLAIN_explainer_o'],
            e['diff_single'],
            e['map_difference'],
            e['k'],
            e['error'],
            e['instance'],
            e['target_class'],
            e['instance_class_index'],
            e['prob']
        ))

    def test_get_explanation_zoo_naive_bayes(self):
        e = get_explanation(join(DEFAULT_DIR, "datasets/zoo.arff"), "sklearn_nb")
        self.assertMatchSnapshot((
            e['XPLAIN_explainer_o'],
            e['diff_single'],
            e['map_difference'],
            e['k'],
            e['error'],
            e['instance'],
            e['target_class'],
            e['instance_class_index'],
            e['prob']
        ))

    def test_get_explanation_adult_naive_bayes(self):
        e = get_explanation(join(DEFAULT_DIR, "datasets/adult_d.arff"), "sklearn_nb")
        self.assertMatchSnapshot((
            e['XPLAIN_explainer_o'],
            e['diff_single'],
            e['map_difference'],
            e['k'],
            e['error'],
            e['instance'],
            e['target_class'],
            e['instance_class_index'],
            e['prob']
        ))
