# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import pickle
import random
# noinspection PyUnresolvedReferences
from collections import Counter
from collections import defaultdict
# noinspection PyUnresolvedReferences
from copy import deepcopy
# noinspection PyUnresolvedReferences
from os import path
# noinspection PyUnresolvedReferences
from os.path import join
from typing import Tuple, List

import Orange
import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import DEFAULT_DIR

MAX_SAMPLE_COUNT = 100
MT = 1


class Dataset:
    _df: pd.DataFrame

    def __init__(self, data, attributes):
        self._df = pd.DataFrame(data)
        self.columns = attributes

        # Rename columns from 0,1,... to the attributes[0,1,...][0]
        columns_mapper = {i: a for (i, a) in enumerate([a for (a, _) in attributes])}
        self._df = self._df.rename(columns=columns_mapper)

        # Encode categorical columns with value between 0 and n_classes-1
        # Keep the columns encoders used to perform the inverse transformation
        # https://stackoverflow.com/a/31939145
        self._column_encoders = defaultdict(LabelEncoder)
        self._encoded_df = self._df.apply(lambda x: self._column_encoders[x.name].fit_transform(x))

    def class_values(self):
        """All possible classes in the dataset"""
        return self.columns[-1][1]

    def X(self):
        """All rows' attributes as a pandas DataFrame."""
        return self._encoded_df.iloc[:, :-1]

    def Y(self):
        """All rows' classes as a pandas Series."""
        return self._encoded_df.iloc[:, -1]

    def X_numpy(self):
        """All rows' attributes as a numpy float64 array."""
        return self._encoded_df.iloc[:, :-1].to_numpy().astype(np.float64)

    def Y_numpy(self):
        """All rows' classes as a numpy float64 array."""
        return self._encoded_df.iloc[:, -1].to_numpy().astype(np.float64)

    def attributes(self):
        return self.columns[:-1]

    def row_inverse_transform_value(self, value, column):
        """"Given a value (one column of a row) and that column's name, return itsdecoded value"""
        return self._column_encoders[column].inverse_transform(value)

    def class_column_name(self):
        """"The column name of the class attribute"""
        return self.columns[-1][0]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, item):
        return self._encoded_df.iloc[item]

    def orange_domain(self):
        """"Return a Orange.data.Domain built using the dataset's attributes"""
        orange_vars = [Orange.data.DiscreteVariable.make(name, vals) for (name, vals) in
                       self.columns]
        return Orange.data.Domain(attributes=orange_vars[:-1], class_vars=orange_vars[-1])

    def to_arff_obj(self):
        obj = {'relation': self.class_column_name(),
               'attributes': self.columns,
               'data': self._df.values.tolist()}
        return obj


def make_orange_instance(dataset, index):
    return Orange.data.Instance(dataset.orange_domain(), dataset[index])


def table_to_arff(t):
    obj = {'relation': t.domain.class_var.name,
           'attributes': [(v.name, v.values) for v in t.domain.variables],
           'data': [list(r.values()) for r in t]}
    return obj


def assert_orange_pd_equal(table: Orange.data.Table, dataset: Dataset):
    # TODO(Andrea): Remove when Orange is completely out
    assert len(table) == len(dataset)


# noinspection PyUnresolvedReferences
def import_dataset(dataset_name: str, explain_indices: List[int], random_explain_dataset: bool) -> \
        Tuple[Tuple[Orange.data.Table, Dataset], Tuple[Orange.data.Table, Dataset], int, List[str]]:
    if dataset_name[-4:] == "arff":
        orange_dataset, pd_dataset = loadARFF(dataset_name)
    else:
        orange_dataset = Orange.data.Table(dataset_name)

        if False in [i.is_discrete for i in orange_dataset[0].domain.attributes]:
            disc = Orange.preprocess.Discretize()
            disc.method = Orange.preprocess.discretize.EqualFreq(3)
            orange_dataset = disc(orange_dataset)

        dataset_file_name = join(DEFAULT_DIR, "datasets", dataset_name) + ".arff"

        with open(dataset_file_name, "w") as f:
            arff.dump(table_to_arff(orange_dataset), f)

        orange_dataset, pd_dataset = loadARFF(dataset_file_name)
        assert_orange_pd_equal(orange_dataset, pd_dataset)

    assert_orange_pd_equal(orange_dataset, pd_dataset)

    dataset_len = len(orange_dataset)
    training_indices = list(range(dataset_len))

    if random_explain_dataset:
        random.seed(1)
        # small dataset
        if dataset_len < (2 * MAX_SAMPLE_COUNT):
            samples = int(0.2 * dataset_len)
        else:
            samples = MAX_SAMPLE_COUNT

        # Randomly pick some instances to remove from the training dataset and use in the
        # explain dataset
        explain_indices = list(random.sample(training_indices, samples))
    for i in explain_indices:
        training_indices.remove(i)

    orange_training_dataset = Orange.data.Table.from_table_rows(orange_dataset, training_indices)
    pd_training_dataset = Dataset(pd_dataset._df.iloc[training_indices], pd_dataset.columns)
    assert_orange_pd_equal(orange_training_dataset, pd_training_dataset)

    orange_explain_dataset = Orange.data.Table.from_table_rows(orange_dataset, explain_indices)
    pd_explain_dataset = Dataset(pd_dataset._df.iloc[explain_indices], pd_dataset.columns)
    assert_orange_pd_equal(orange_explain_dataset, pd_explain_dataset)

    return (orange_training_dataset, pd_training_dataset), (
        orange_explain_dataset, pd_explain_dataset), len(
        orange_training_dataset), \
           [str(i) for i in explain_indices]


def import_datasets(dataset_name: str, explain_indices: List[int],
                    random_explain_dataset: bool) -> Tuple[
    Tuple[Orange.data.Table, Dataset], Tuple[Orange.data.Table, Dataset], int, List[str]]:
    """
    :param dataset_name: path of the dataset file
    :param explain_indices: indices of the instances to be added in the explain dataset
    :param random_explain_dataset: create the explain dataset randomly, will make `explain_idnices`
    futile
    :return:
    """
    assert (dataset_name[-4:] == "arff")

    explain_dataset_name = dataset_name[:-5] + "_explain.arff"

    orange_training_dataset, pd_training_dataset = loadARFF(dataset_name)
    orange_explain_dataset, pd_explain_dataset = loadARFF(explain_dataset_name)
    assert_orange_pd_equal(orange_training_dataset, pd_training_dataset)

    len_dataset = len(orange_training_dataset)
    len_explain_dataset = len(orange_explain_dataset)

    if random_explain_dataset:
        random.seed(7)
        explain_indices = list(random.sample(range(len_explain_dataset), 300))

    orange_explain_dataset = Orange.data.Table.from_table_rows(orange_explain_dataset,
                                                               explain_indices)
    pd_explain_dataset = Dataset(pd_explain_dataset._df.iloc[explain_indices],
                                 pd_explain_dataset.columns)
    assert_orange_pd_equal(orange_explain_dataset, pd_explain_dataset)

    return (orange_training_dataset, pd_training_dataset), (
        orange_explain_dataset, pd_explain_dataset), len_dataset, [str(i) for i in explain_indices]


def loadARFF(filename: str) -> Tuple[Orange.data.Table, Dataset]:
    with open(filename, 'r') as f:
        a = arff.load(f)
        dataset = Dataset(a['data'], a['attributes'])

        table = Orange.data.Table.from_numpy(dataset.orange_domain(), dataset.X_numpy(),
                                             dataset.Y_numpy())
        table.name = a['relation']

        assert_orange_pd_equal(table, dataset)

        return table, dataset


def get_features_names(classifier):
    features_names = []
    for i in range(0, len(classifier.domain.attributes)):
        if ">" in classifier.domain.attributes[i].name:
            features_names.append(
                classifier.domain.attributes[i].name.replace(">", "gr"))

        elif "<" in classifier.domain.attributes[i].name:
            features_names.append(
                classifier.domain.attributes[i].name.replace("<", "low"))
        else:
            features_names.append(classifier.domain.attributes[i].name)

    return features_names


# noinspection PyUnresolvedReferences
def get_classifier(training_datasets: Tuple[Orange.data.Table, Dataset], classifier_name: str,
                   classifier_parameter: str, should_exit) -> Orange.classification.Learner:
    # TODO(Andrea): FINISH continue threading through the pandas dataset
    classifier_name = classifier_name
    classifier = None

    training_dataset, pd_training_dataset = training_datasets

    assert_orange_pd_equal(training_dataset, pd_training_dataset)

    if classifier_name == "tree":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnertree = Orange.classification.SklTreeLearner(
            preprocessors=continuizer, max_depth=7, min_samples_split=5,
            min_samples_leaf=3, random_state=1)
        classifier = learnertree(training_dataset)

    elif classifier_name == "nb":
        learnernb = Orange.classification.NaiveBayesLearner()
        classifier = learnernb(training_dataset)

    elif classifier_name == "nn":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnernet = Orange.classification.NNClassificationLearner(
            preprocessors=continuizer, random_state=42,
            max_iter=1000)
        classifier = learnernet(training_dataset)

    elif classifier_name == "rf":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnerrf = Orange.classification.RandomForestLearner(
            preprocessors=continuizer, random_state=42)
        classifier = learnerrf(training_dataset)

    elif classifier_name == "svm":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnerrf = Orange.classification.SVMLearner(preprocessors=continuizer)
        classifier = learnerrf(training_dataset)

    elif classifier_name == "knn":
        if classifier_parameter is None:
            raise ValueError("k - missing the K parameter")
        elif len(classifier_parameter.split("-")) == 1:
            KofKNN = int(classifier_parameter.split("-")[0])
            distance = ""
        else:
            KofKNN = int(classifier_parameter.split("-")[0])
            distance = classifier_parameter.split("-")[1]
        if not should_exit:
            if distance == "eu":
                metricKNN = 'euclidean'
            elif distance == "ham":
                metricKNN = 'hamming'
            elif distance == "man":
                metricKNN = 'manhattan'
            elif distance == "max":
                metricKNN = 'maximal'
            else:
                metricKNN = 'euclidean'
            continuizer = Orange.preprocess.Continuize()
            continuizer.multinomial_treatment = continuizer.Indicators
            knnLearner = Orange.classification.KNNLearner(
                preprocessors=continuizer, n_neighbors=KofKNN,
                metric=metricKNN, weights='uniform', algorithm='auto',
                metric_params=None)
            classifier = knnLearner(training_dataset)
    else:
        raise ValueError("Classifier not available")

    return classifier


def gen_neighbors_info(training_dataset: Dataset, nbrs,
                       instance: Orange.data.Instance, k: int,
                       unique_filename: str, classifier):
    nearest_neighbors_ixs = nbrs.kneighbors([instance.x], k,
                                            return_distance=False)[0]
    orange_closest_instance = make_orange_instance(training_dataset, nearest_neighbors_ixs[0])

    classified_instance = deepcopy(instance)
    classified_instance.set_class(classifier(instance)[0])
    classified_instances = [classified_instance]

    for neigh_ix in nearest_neighbors_ixs:
        orange_neigh = make_orange_instance(training_dataset, neigh_ix)
        classified_neigh = deepcopy(orange_neigh)
        classified_neigh.set_class(classifier(orange_neigh)[0])

        classified_instances.append(classified_neigh)

    pd_classified_instances_dataset = Dataset(
        [c.list for c in classified_instances],
        training_dataset.columns)

    closest_instance_classified = deepcopy(orange_closest_instance)
    closest_instance_classified.set_class(classifier(orange_closest_instance)[0])
    pd_closest_instance_dataset = Dataset(
        [c.list for c in [closest_instance_classified]],
        training_dataset.columns)

    p = DEFAULT_DIR + unique_filename
    if not os.path.exists(p):
        os.makedirs(p)

    with open(join(p, "Knnres.arff"), "w") as f:
        arff.dump(pd_classified_instances_dataset.to_arff_obj(), f)
    with open(join(p, "Filetest.arff"), "w") as f:
        arff.dump(pd_closest_instance_dataset.to_arff_obj(), f)


def get_relevant_subset_from_local_rules(impo_rules, oldinputAr):
    inputAr = []
    iA = []
    nInputAr = []

    for i2 in range(0, len(impo_rules)):
        intInputAr = []
        val = impo_rules[i2].split(",")
        for i3 in range(0, len(val)):
            intInputAr.append(int(val[i3]))
            iA.append(int(val[i3]))
        nInputAr.append(intInputAr)
    iA2 = list(sorted(set(iA)))
    inputAr.append(iA2)
    if inputAr[0] not in nInputAr:
        nInputAr.append(inputAr[0])
    inputAr = deepcopy(nInputAr)
    oldAr_set = set(map(tuple, oldinputAr))
    # In order to not recompute the prior probability of a Subset again
    newInputAr = [x for x in inputAr if tuple(x) not in oldAr_set]
    oldAr_set = set(map(tuple, oldinputAr))

    return inputAr, nInputAr, newInputAr, oldAr_set


def compute_prediction_difference_subset(training_dataset_,
                                         instance,
                                         rule_body_indices,
                                         classifier,
                                         instance_class_index,
                                         instance_predictions_cache):
    """
    Compute the prediction difference for an instance in a training_dataset, w.r.t. some
    rules and a class, given a classifier
    """
    training_dataset: Dataset = training_dataset_[MT]

    rule_attributes = [
        list(training_dataset.attributes())[rule_body_index - 1][0] for
        rule_body_index in rule_body_indices]

    # Take only the considered attributes from the dataset
    filtered_dataset = training_dataset.X()[rule_attributes]

    # Count how many times a set of attribute values appears in the dataset
    attribute_sets_occurrences = dict(
        Counter(map(tuple, filtered_dataset.values.tolist())).items())

    # For each set of attributes
    differences = [compute_perturbed_difference(item, classifier, instance, instance_class_index,
                                                rule_attributes, training_dataset_) for
                   item in
                   attribute_sets_occurrences.items()]

    prediction_difference = sum(differences)

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = classifier(instance, True)[0][instance_class_index]
    prediction_differences = p - prediction_difference

    return prediction_differences


def compute_perturbed_difference(item, classifier, instance, instance_class_index,
                                 rule_attributes, training_dataset_):
    (attribute_set, occurrences) = item
    perturbed_instance = deepcopy(instance)
    for i in range(len(rule_attributes)):
        perturbed_instance[rule_attributes[i]] = attribute_set[i]
    # cache_key = tuple(perturbed_instance.x)
    # if cache_key not in instance_predictions_cache:
    #     instance_predictions_cache[cache_key] = classifier(perturbed_instance, True)[0][
    #         instance_class_index]
    # prob = instance_predictions_cache[cache_key]
    prob = classifier(perturbed_instance, True)[0][instance_class_index]

    # Compute the prediction difference using the weighted average of the
    # probability over the frequency of this attribute set in the
    # dataset
    difference = prob * occurrences / len(training_dataset_[MT])
    return difference


# Single explanation. Change 1 value at the time e compute the difference
def compute_prediction_difference_single(instance, classifier, target_class_index,
                                         training_dataset_):
    training_dataset = training_dataset_[MT]
    dataset_len = len(training_dataset)

    attribute_pred_difference = [0] * len(training_dataset.attributes())

    # The probability of `instance` belonging to class `target_class_index`
    prob = classifier(instance, True)[0][target_class_index]

    # For each `instance` attribute
    for (attr_ix, (attr, _)) in enumerate(training_dataset.attributes()):
        # Create a dataset containing only the column of the attribute
        filtered_dataset = training_dataset.X()[attr]

        # Count how many times each value of that attribute appears in the dataset
        attr_occurrences = dict(Counter(filtered_dataset).items())

        # For each value of the attribute
        for attr_val in attr_occurrences:
            # Create an instance whose attribute `attr` has that value (`attr_val`)
            perturbed_instance = deepcopy(instance)
            perturbed_instance[attr] = attr_val

            # See how the prediction changes
            prob = classifier(perturbed_instance, True)[0][target_class_index]

            # Update the attribute difference weighting the prediction by the value frequency
            weight = attr_occurrences[attr_val] / dataset_len
            difference = prob * weight
            attribute_pred_difference[attr_ix] += difference

    for i in range(len(attribute_pred_difference)):
        attribute_pred_difference[i] = prob - attribute_pred_difference[i]

    return attribute_pred_difference


def compute_error_approximation(mappa_class, pred, out_data, impo_rules_complete, classname,
                                map_difference):
    PI = pred - mappa_class[classname]
    Sum_Deltas = sum(out_data)
    # UPDATED_EP
    impo_rules_completeC = ", ".join(map(str, list(max(impo_rules_complete, key=len))))

    approx_single_d = abs(PI - Sum_Deltas)
    approx_single_rel = approx_single_d / abs(PI)

    if impo_rules_completeC != "":
        if len(impo_rules_completeC.replace(" ", "").split(",")) > 1:
            Delta_impo_rules_completeC = map_difference[impo_rules_completeC.replace(" ", "")]
            PI_approx2 = Delta_impo_rules_completeC
            Sum_Deltas_not_in = 0.0
            # Sum of delta_i for each attribute not included
            for i_out_data in range(0, len(out_data)):
                if str(i_out_data + 1) not in impo_rules_completeC.replace(" ", "").split(","):
                    Sum_Deltas_not_in = Sum_Deltas_not_in + out_data[i_out_data]
        else:
            index = int(impo_rules_completeC.replace(" ", "").split(",")[0]) - 1
            PI_approx2 = out_data[index]
        approx2 = abs(PI - PI_approx2)
        approx_rel2 = approx2 / abs(PI)
    else:
        PI_approx2 = 0.0
        approx_rel2 = 1

    approx2 = abs(PI - PI_approx2)

    return approx_single_rel, approx2, approx_rel2


def getStartKValueSimplified(len_dataset):
    if len_dataset < 150:
        maxN = len_dataset
    elif len_dataset < 1000:
        maxN = int(len_dataset / 2)
    elif len_dataset < 10000:
        maxN = int(len_dataset / 10)
    else:
        maxN = int(len_dataset * 5 / 100)
    return maxN


def compute_class_frequency(data_):
    pd_data = data_[MT]

    class_frequency = {}
    h = len(pd_data)

    for i in range(h):
        row = pd_data[i]
        cc = pd_data.class_column_name()
        row_class = pd_data.row_inverse_transform_value(row[cc], cc)
        if row_class in class_frequency:
            class_frequency[row_class] = class_frequency[row_class] + 1.0
        else:
            class_frequency[row_class] = 1.0

    for key in class_frequency.keys():
        class_frequency[key] = class_frequency[key] / h

    return class_frequency


def convert_orange_table_to_pandas(orangeTable, ids=None, sel="all", cl=None, mapName=None):
    if sel == "all":
        dataK = [orangeTable[k].list for k in range(0, len(orangeTable))]
    else:
        dataK = [orangeTable[k].list for k in sel]

    columnsA = [i.name for i in orangeTable.domain.variables]

    if orangeTable.domain.metas != ():
        for i in range(0, len(orangeTable.domain.metas)):
            columnsA.append(orangeTable.domain.metas[i].name)
    data = pd.DataFrame(data=dataK, columns=columnsA)

    if cl is not None and sel != "all" and mapName is not None:
        y_pred = [mapName[cl(orangeTable[k], False)[0]] for k in sel]
        data["pred"] = y_pred

    if ids is not None:
        data["instance_id"] = ids
        data = data.set_index('instance_id')

    return data


def savePickle(model, dirO, name):
    os.makedirs(dirO)
    with open(dirO + "/" + name + '.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def openPickle(dirO, name):
    if path.exists(dirO + "/" + name + '.pickle'):
        with open(dirO + "/" + name + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return False


def get_KNN_threshold_max(KneighborsUser, len_dataset, thresholdError,
                          maxKNNUser):
    if KneighborsUser:
        k = int(KneighborsUser)
    else:
        import math
        k = int(round(math.sqrt(len_dataset)))

    if thresholdError:
        threshold = float(thresholdError)
    else:
        threshold = 0.10

    if maxKNNUser:
        max_n = int(maxKNNUser)
    else:
        max_n = getStartKValueSimplified(len_dataset)

    return k, threshold, max_n
