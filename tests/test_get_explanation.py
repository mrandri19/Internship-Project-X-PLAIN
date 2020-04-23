import random
from os.path import join
from typing import Tuple, List
from unittest import TestCase

import arff

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
    explainer = XPLAIN_explainer(clf, train_dataset)

    instance = explain_dataset.get_decoded(0)

    cc = explain_dataset.class_column_name()
    target_class_index = instance[cc]

    return explainer.explain_instance(explain_dataset[0], target_class_index)


class TestGet_explanation(TestCase):
    def test_get_explanation_zoo_random_forest(self):
        e = get_explanation(join(DEFAULT_DIR, "datasets/zoo.arff"), "sklearn_rf")
        self.assertEqual((
            e['diff_single'],
            e['map_difference'],
            e['k'],
            e['error'],
            e['target_class'],
            e['instance_class_index'],
            e['prob']
        ), (
            [
                0.11024691358024685,
                0.02308641975308645,
                0.19728395061728388,
                0.31407407407407417,
                0.004938271604938316,
                0.006913580246913575,
                0.0,
                0.07111111111111112,
                0.00864197530864197,
                0.03358024691358019,
                0.0007407407407408195,
                0.0,
                0.005185185185185182,
                0.0,
                0.0,
                0.0
            ],
            {
                '1,2,3,4,8,9,10,11': 0.5839506172839506
            },
            33,
            0.00864197530864197,
            'mammal',
            5,
            1.0
        ))

    def test_get_explanation_zoo_naive_bayes(self):
        e = get_explanation(join(DEFAULT_DIR, "datasets/zoo.arff"), "sklearn_nb")
        self.assertEqual((
            e['diff_single'],
            e['map_difference'],
            e['k'],
            e['error'],
            e['target_class'],
            e['instance_class_index'],
            e['prob']
        ), (
            [
                0.0018493244327110192,
                0.0005835022761034869,
                0.04128880084184605,
                0.002583590538021041,
                -0.002213184871231788,
                -0.0005848709570587252,
                -0.0026459469615073283,
                -0.0027706378431165968,
                -0.0033543844546455315,
                -0.0033771583400726835,
                -0.0023206657800494135,
                -0.0032075223689916887,
                -0.0027332607152817934,
                -0.003227422277876868,
                -0.0035048571858332656,
                -0.0015305424415449354
            ],
            {
                '2,3,4,8,9,10,11': 0.39514075431687523
            },
            42,
            0.19470029326431426,
            'mammal',
            5,
            0.9972484549885969
        ))

    def test_get_explanation_adult_naive_bayes(self):
        e = get_explanation(join(DEFAULT_DIR, "datasets/adult_d.arff"), "sklearn_nb")
        self.assertEqual((
            e['diff_single'],
            e['map_difference'],
            e['k'],
            e['error'],
            e['target_class'],
            e['instance_class_index'],
            e['prob']
        ), (
            [
                0.04267021431143747,
                0.0013601660256593595,
                -0.00025793916220828716,
                -0.002583202995719347,
                0.41800765869397466,
                -0.07252560975447497,
                0.038882713120766854,
                -0.0002026726258052003,
                6.138967731539324e-05,
                0.01974576342296308,
                -0.013082534026229053
            ],
            {
                '1,10,11': 0.04904249942186634,
                '1,3,5,6,8,9,10,11': 0.32111161850226366,
                '1,3,8,9': 0.04214440558178556,
                '1,6': -0.044924825486333586,
                '5': 0.41800765869397466
            },
            118,
            0.01442188127852051,
            '<=50K',
            0,
            0.8066897372237432
        ))
