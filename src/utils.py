# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import pickle
import random
# noinspection PyUnresolvedReferences
from collections import Counter, defaultdict
# noinspection PyUnresolvedReferences
from copy import deepcopy
# noinspection PyUnresolvedReferences
from os import path
# noinspection PyUnresolvedReferences
from os.path import join
from typing import Tuple, List

import Orange
import arff
# noinspection PyUnresolvedReferences
import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import sklearn

from src import DEFAULT_DIR
from src.dataset import Dataset

MAX_SAMPLE_COUNT = 100
MT = 1


def make_orange_instance_index(dataset: Dataset, index: int) -> Orange.data.Instance:
    """"Make an Orange.data.Instance from the row at `index` of the `dataset`"""
    return Orange.data.Instance(dataset.orange_domain(), dataset[index])


def make_orange_instance(dataset: Dataset, instance: pd.Series) -> Orange.data.Instance:
    """"Make an Orange.data.Instance from an `instance` (a row) of the `dataset`"""
    return Orange.data.Instance(dataset.orange_domain(), instance)


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
        Tuple[Dataset, Dataset, int, List[str]]:
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

        pd_dataset = loadARFF(dataset_file_name)

    dataset_len = len(pd_dataset)
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

    pd_training_dataset = Dataset(pd_dataset._decoded_df.iloc[training_indices], pd_dataset.columns)

    pd_explain_dataset = Dataset(pd_dataset._decoded_df.iloc[explain_indices], pd_dataset.columns)

    return pd_training_dataset, pd_explain_dataset, len(
        pd_training_dataset), \
           [str(i) for i in explain_indices]


def import_datasets(dataset_name: str, explain_indices: List[int],
                    random_explain_dataset: bool) -> Tuple[
    Dataset, Dataset, int, List[str]]:
    """
    :param dataset_name: path of the dataset file
    :param explain_indices: indices of the instances to be added in the explain dataset
    :param random_explain_dataset: randomly sample 300 rows from the _explain.arff file to make the
                                   explain, will make `explain_indices` futile
    :return:
    """
    assert (dataset_name[-4:] == "arff")

    explain_dataset_name = dataset_name[:-5] + "_explain.arff"

    pd_training_dataset = loadARFF(dataset_name)
    pd_explain_dataset = loadARFF(explain_dataset_name)

    len_dataset = len(pd_training_dataset)
    len_explain_dataset = len(pd_explain_dataset)

    if random_explain_dataset:
        random.seed(7)
        explain_indices = list(random.sample(range(len_explain_dataset), 300))

    pd_explain_dataset = Dataset(pd_explain_dataset._decoded_df.iloc[explain_indices],
                                 pd_explain_dataset.columns)

    return pd_training_dataset, pd_explain_dataset, len_dataset, [str(i) for i in
                                                                  explain_indices]


def loadARFF(filename: str) -> Dataset:
    with open(filename, 'r') as f:
        a = arff.load(f)
        dataset = Dataset(a['data'], a['attributes'])

        return dataset


# noinspection PyUnresolvedReferences
def get_classifier(training_dataset: Dataset, classifier_name: str,
                   classifier_parameter: str) -> Tuple[Orange.classification.Learner, object]:
    classifier_name = classifier_name

    orange_training_dataset = training_dataset.to_orange_table()

    if classifier_name == "tree":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnertree = Orange.classification.SklTreeLearner(
            preprocessors=continuizer, max_depth=7, min_samples_split=5,
            min_samples_leaf=3, random_state=1)
        orange_clf = learnertree(orange_training_dataset)
        raise NotImplementedError

    elif classifier_name == "nb":
        from sklearn.naive_bayes import MultinomialNB

        learnernb = Orange.classification.NaiveBayesLearner()
        orange_clf = learnernb(orange_training_dataset)
        skl_clf = MultinomialNB().fit(orange_training_dataset.X, orange_training_dataset.Y)

        return orange_clf, skl_clf

    elif classifier_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators

        learnerrf = Orange.classification.RandomForestLearner(
            preprocessors=continuizer, random_state=42)
        orange_clf = learnerrf(orange_training_dataset)
        skl_clf = RandomForestClassifier().fit(orange_training_dataset.X, orange_training_dataset.Y)

        return orange_clf, skl_clf

    elif classifier_name == "nn":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnernet = Orange.classification.NNClassificationLearner(
            preprocessors=continuizer, random_state=42,
            max_iter=1000)
        orange_clf = learnernet(orange_training_dataset)
        raise NotImplementedError

    elif classifier_name == "svm":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnerrf = Orange.classification.SVMLearner(preprocessors=continuizer)
        orange_clf = learnerrf(orange_training_dataset)
        raise NotImplementedError

    elif classifier_name == "knn":

        if classifier_parameter is None:
            raise ValueError("k - missing the K parameter")
        elif len(classifier_parameter.split("-")) == 1:
            KofKNN = int(classifier_parameter.split("-")[0])
            distance = ""
        else:
            KofKNN = int(classifier_parameter.split("-")[0])
            distance = classifier_parameter.split("-")[1]

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
        orange_clf = knnLearner(orange_training_dataset)
        raise NotImplementedError
    else:
        raise ValueError("Classifier not available")


def gen_neighbors_info(training_dataset: Dataset, nbrs,
                       instance: Orange.data.Instance, k: int,
                       unique_filename: str, classifier):
    nearest_neighbors_ixs = nbrs.kneighbors([instance.x], k,
                                            return_distance=False)[0]
    orange_closest_instance = make_orange_instance_index(training_dataset, nearest_neighbors_ixs[0])

    classified_instance = deepcopy(instance)
    classified_instance.set_class(
        classifier[MT].predict(instance.x.reshape(1, -1))[0].astype(np.int64))
    classified_instances = [classified_instance]

    for neigh_ix in nearest_neighbors_ixs:
        orange_neigh = make_orange_instance_index(training_dataset, neigh_ix)
        classified_neigh = deepcopy(orange_neigh)
        classified_neigh.set_class(
            classifier[MT].predict(orange_neigh.x.reshape(1, -1))[0].astype(np.int64))

        classified_instances.append(classified_neigh)

    pd_classified_instances_dataset = Dataset(
        [c.list for c in classified_instances],
        training_dataset.columns)

    closest_instance_classified = deepcopy(orange_closest_instance)
    closest_instance_classified.set_class(
        classifier[MT].predict(orange_closest_instance.x.reshape(1, -1))[0].astype(np.int64))
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


def compute_prediction_difference_subset(training_dataset: Dataset,
                                         instance: Orange.data.Instance,
                                         rule_body_indices,
                                         classifier,
                                         instance_class_index,
                                         instance_predictions_cache):
    """
    Compute the prediction difference for an instance in a training_dataset, w.r.t. some
    rules and a class, given a classifier
    """

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
                                                rule_attributes, training_dataset) for
                   item in
                   attribute_sets_occurrences.items()]

    prediction_difference = sum(differences)

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = classifier[MT].predict_proba(instance.x.reshape(1, -1))[0][instance_class_index]
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
    prob = classifier[MT].predict_proba(perturbed_instance.x.reshape(1, -1))[0][
        instance_class_index]

    # Compute the prediction difference using the weighted average of the
    # probability over the frequency of this attribute set in the
    # dataset
    difference = prob * occurrences / len(training_dataset_)
    return difference


# Single explanation. Change 1 value at the time e compute the difference
def compute_prediction_difference_single(instance, classifier, target_class_index,
                                         training_dataset_):
    training_dataset = training_dataset_
    dataset_len = len(training_dataset)

    attribute_pred_difference = [0] * len(training_dataset.attributes())

    # The probability of `instance` belonging to class `target_class_index`
    orange_prob = classifier[MT].predict_proba(instance.x.reshape(1, -1))[0][target_class_index]

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
            orange_prob = classifier[MT].predict_proba(perturbed_instance.x.reshape(1, -1))[0][
                target_class_index]

            # Update the attribute difference weighting the prediction by the value frequency
            weight = attr_occurrences[attr_val] / dataset_len
            difference = orange_prob * weight
            attribute_pred_difference[attr_ix] += difference

    for i in range(len(attribute_pred_difference)):
        attribute_pred_difference[i] = orange_prob - attribute_pred_difference[i]

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


def compute_class_frequency(dataset: Dataset):
    class_frequency = {}
    h = len(dataset)

    for i in range(h):
        row = dataset[i]
        cc = dataset.class_column_name()
        row_class = dataset.row_inverse_transform_value(row[cc], cc)
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
