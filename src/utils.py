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

import arff
# noinspection PyUnresolvedReferences
import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import sklearn

from src import DEFAULT_DIR
from src.dataset import Dataset

MAX_SAMPLE_COUNT = 100


def table_to_arff(t):
    obj = {'relation': t.domain.class_var.name,
           'attributes': [(v.name, v.values) for v in t.domain.variables],
           'data': [list(r.values()) for r in t]}
    return obj


# noinspection PyUnresolvedReferences
def import_dataset(dataset_name: str, explain_indices: List[int], random_explain_dataset: bool) -> \
        Tuple[Dataset, Dataset, int, List[str]]:
    assert dataset_name.endswith("arff")

    dataset = load_arff(dataset_name)

    dataset_len = len(dataset)
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

    training_dataset = Dataset(dataset._decoded_df.iloc[training_indices], dataset.columns)

    explain_dataset = Dataset(dataset._decoded_df.iloc[explain_indices], dataset.columns)

    return training_dataset, explain_dataset, len(training_dataset), \
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

    pd_training_dataset = load_arff(dataset_name)
    pd_explain_dataset = load_arff(explain_dataset_name)

    len_dataset = len(pd_training_dataset)
    len_explain_dataset = len(pd_explain_dataset)

    if random_explain_dataset:
        random.seed(7)
        explain_indices = list(random.sample(range(len_explain_dataset), 300))

    pd_explain_dataset = Dataset(pd_explain_dataset._decoded_df.iloc[explain_indices],
                                 pd_explain_dataset.columns)

    return pd_training_dataset, pd_explain_dataset, len_dataset, [str(i) for i in
                                                                  explain_indices]


def load_arff(filename: str) -> Dataset:
    with open(filename, 'r') as f:
        a = arff.load(f)
        dataset = Dataset(a['data'], a['attributes'])

        return dataset


# noinspection PyUnresolvedReferences
def get_classifier(training_dataset: Dataset, classifier_name: str) -> sklearn.base.ClassifierMixin:
    if classifier_name == "sklearn_nb":
        from sklearn.naive_bayes import MultinomialNB

        skl_clf = MultinomialNB().fit(training_dataset.X_numpy(), training_dataset.Y_numpy())

        return skl_clf

    elif classifier_name == "sklearn_rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        pipe = make_pipeline(OneHotEncoder(), RandomForestClassifier(random_state=42))
        skl_clf = pipe.fit(training_dataset.X_numpy(), training_dataset.Y_numpy())

        return skl_clf

    elif classifier_name == "nn_label_enc":
        from sklearn.neural_network import MLPClassifier

        skl_clf = MLPClassifier(random_state=42, max_iter=1000).fit(training_dataset.X_numpy(),
                                                                    training_dataset.Y_numpy())
        return skl_clf

    elif classifier_name == "nn_onehot_enc":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        pipe = make_pipeline(OneHotEncoder(), MLPClassifier(random_state=42, max_iter=1000))
        skl_clf = pipe.fit(training_dataset.X_numpy(), training_dataset.Y_numpy())

        return skl_clf

    else:
        raise ValueError("Classifier not available")


def gen_neighbors_info(training_dataset: Dataset, nbrs,
                       encoded_instance: pd.Series, k: int,
                       unique_filename: str, clf):
    cc = training_dataset.class_column_name()
    instance_x = encoded_instance[:-1].to_numpy()

    nearest_neighbors_ixs = nbrs.kneighbors([instance_x], k,
                                            return_distance=False)[0]

    classified_instance = deepcopy(encoded_instance)
    classified_instance[cc] = clf.predict(instance_x.reshape(1, -1))[0]
    classified_instances = [classified_instance]

    for neigh_ix in nearest_neighbors_ixs:
        neigh = training_dataset[neigh_ix]
        neigh_x = neigh[:-1].to_numpy()

        classified_neigh = deepcopy(neigh)
        classified_neigh[cc] = clf.predict(neigh_x.reshape(1, -1))[0]

        classified_instances.append(classified_neigh)

    classified_instances_dataset = Dataset(
        [training_dataset.inverse_transform_instance(c) for c in classified_instances],
        training_dataset.columns)

    closest_instance = training_dataset[nearest_neighbors_ixs[0]]
    closest_instance_x = closest_instance[:-1].to_numpy()

    closest_instance_classified = deepcopy(closest_instance)
    closest_instance_classified[cc] = clf.predict(closest_instance_x.reshape(1, -1))[0]

    closest_instance_dataset = Dataset(
        [training_dataset.inverse_transform_instance(closest_instance_classified)],
        training_dataset.columns)

    p = DEFAULT_DIR + unique_filename
    if not os.path.exists(p):
        os.makedirs(p)

    with open(join(p, "Knnres.arff"), "w") as f:
        arff.dump(classified_instances_dataset.to_arff_obj(), f)
    with open(join(p, "Filetest.arff"), "w") as f:
        arff.dump(closest_instance_dataset.to_arff_obj(), f)


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
                                         encoded_instance: pd.Series,
                                         rule_body_indices,
                                         clf,
                                         instance_class_index,
                                         instance_predictions_cache):
    """
    Compute the prediction difference for an instance in a training_dataset, w.r.t. some
    rules and a class, given a classifier
    """

    encoded_instance_x = encoded_instance[:-1].to_numpy()

    rule_attributes = [
        list(training_dataset.attributes())[rule_body_index - 1][0] for
        rule_body_index in rule_body_indices]

    # Take only the considered attributes from the dataset
    filtered_dataset = training_dataset.X()[rule_attributes]

    # Count how many times a set of attribute values appears in the dataset
    attribute_sets_occurrences = dict(
        Counter(map(tuple, filtered_dataset.values.tolist())).items())

    # For each set of attributes
    differences = [
        compute_perturbed_difference(item, clf, encoded_instance,
                                     instance_class_index,
                                     rule_attributes, training_dataset) for
        item in
        attribute_sets_occurrences.items()]

    prediction_difference = sum(differences)

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = clf.predict_proba(encoded_instance_x.reshape(1, -1))[0][instance_class_index]
    prediction_differences = p - prediction_difference

    return prediction_differences


def compute_perturbed_difference(item, clf, encoded_instance,
                                 instance_class_index,
                                 rule_attributes, training_dataset_):
    (attribute_set, occurrences) = item

    perturbed_instance = deepcopy(encoded_instance)
    for i in range(len(rule_attributes)):
        perturbed_instance[rule_attributes[i]] = attribute_set[i]

    # cache_key = tuple(perturbed_instance.x)
    # if cache_key not in instance_predictions_cache:
    #     instance_predictions_cache[cache_key] = classifier(perturbed_instance, True)[0][
    #         instance_class_index]
    # prob = instance_predictions_cache[cache_key]

    prob = clf.predict_proba(perturbed_instance[:-1].to_numpy().reshape(1, -1))[0][
        instance_class_index]

    # Compute the prediction difference using the weighted average of the
    # probability over the frequency of this attribute set in the
    # dataset
    difference = prob * occurrences / len(training_dataset_)
    return difference


# Single explanation. Change 1 value at the time e compute the difference
def compute_prediction_difference_single(encoded_instance, clf, target_class_index,
                                         training_dataset):
    dataset_len = len(training_dataset)

    encoded_instance_x = encoded_instance[:-1].to_numpy()

    attribute_pred_difference = [0] * len(training_dataset.attributes())

    # The probability of `instance` belonging to class `target_class_index`
    class_prob = clf.predict_proba(encoded_instance_x.reshape(1, -1))[0][
        target_class_index]

    # For each `instance` attribute
    for (attr_ix, (attr, _)) in enumerate(training_dataset.attributes()):
        # Create a dataset containing only the column of the attribute
        filtered_dataset = training_dataset.X()[attr]

        # Count how many times each value of that attribute appears in the dataset
        attr_occurrences = dict(Counter(filtered_dataset).items())

        # For each value of the attribute
        for attr_val in attr_occurrences:
            # Create an instance whose attribute `attr` has that value (`attr_val`)
            perturbed_encoded_instance = deepcopy(encoded_instance)
            perturbed_encoded_instance[attr] = attr_val
            perturbed_encoded_instance_x = perturbed_encoded_instance[:-1].to_numpy()

            # See how the prediction changes
            class_prob = \
                clf.predict_proba(perturbed_encoded_instance_x.reshape(1, -1))[0][
                    target_class_index]

            # Update the attribute difference weighting the prediction by the value frequency
            weight = attr_occurrences[attr_val] / dataset_len
            difference = class_prob * weight
            attribute_pred_difference[attr_ix] += difference

    for i in range(len(attribute_pred_difference)):
        attribute_pred_difference[i] = class_prob - attribute_pred_difference[i]

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
        return len_dataset
    elif len_dataset < 1000:
        return int(len_dataset / 2)
    elif len_dataset < 10000:
        return int(len_dataset / 10)
    else:
        return int(len_dataset * 5 / 100)


def compute_class_frequency(dataset: Dataset):
    class_frequency = defaultdict(float)

    for row in dataset.Y_decoded():
        class_frequency[row] += 1.0

    for key in class_frequency.keys():
        class_frequency[key] /= len(dataset)

    return class_frequency


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
