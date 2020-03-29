# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
import itertools
import os
# noinspection PyUnresolvedReferences
import shutil
# noinspection PyUnresolvedReferences
import subprocess
# noinspection PyUnresolvedReferences
import uuid
# noinspection PyUnresolvedReferences
from collections import Counter, defaultdict
# noinspection PyUnresolvedReferences
from copy import deepcopy
# noinspection PyUnresolvedReferences
from os import path
# noinspection PyUnresolvedReferences
from os.path import join
# noinspection PyUnresolvedReferences
from typing import Tuple

# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
import sklearn.neighbors
from l3wrapper.l3wrapper import L3Classifier

# noinspection PyUnresolvedReferences
from src import DEFAULT_DIR
# noinspection PyUnresolvedReferences
from src.dataset import Dataset

ERROR_DIFFERENCE_THRESHOLD = 0.01
TEMPORARY_FOLDER_NAME = "tmp"
ERROR_THRESHOLD = 0.02
MINIMUM_SUPPORT = 0.01


class XPLAIN_explainer:
    def __init__(self, clf, train_dataset, min_sup=MINIMUM_SUPPORT):
        self.unique_filename = os.path.join(TEMPORARY_FOLDER_NAME, str(uuid.uuid4()))

        self.clf = clf

        self.train_dataset = train_dataset

        self.K, _, self.max_K = get_KNN_threshold_max(None, len(self.train_dataset), None, None)
        self.starting_K = self.K

        self.nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=len(self.train_dataset), metric='euclidean',
            algorithm='auto', metric_params=None).fit(
            self.train_dataset.X_numpy())

        self.min_sup = min_sup

        self.decoded_class_frequencies = self.train_dataset.Y_decoded().value_counts(normalize=True)

    def explain_instance(self, encoded_instance: pd.Series, decoded_target_class):
        target_class_index = self.train_dataset.class_values().index(decoded_target_class)

        encoded_instance_x = encoded_instance[:-1].to_numpy().reshape(1, -1)

        # Problem with very small training dataset. The starting k is low, very few examples:
        # difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting k: proportional to the class frequence
        small_dataset_len = 150
        training_dataset_len = len(self.train_dataset)
        if training_dataset_len < small_dataset_len:
            decoded_pred_class = self.train_dataset.class_values()[
                self.clf.predict(encoded_instance_x)[0].astype(int)]
            self.starting_K = max(
                int(self.decoded_class_frequencies[decoded_pred_class] * training_dataset_len),
                self.starting_K)

        # Initialize k and error to be defined in case the for loop is not entered
        k = self.starting_K
        old_error = 10.0
        error = 1e9
        class_prob = self.clf.predict_proba(encoded_instance_x)[0][
            target_class_index]
        single_attribute_differences = compute_prediction_difference_single(
            encoded_instance,
            self.clf,
            class_prob,
            target_class_index,
            self.train_dataset)
        difference_map = {}

        first_iteration = True

        errors = []

        # Euristically search for the best k to use to approximate the local model
        for k in range(self.starting_K, self.max_K, self.K):
            print(f"compute_lace_step k={k}")
            difference_map, \
            error = self.compute_lace_step(
                encoded_instance,
                k,
                self.decoded_class_frequencies[decoded_target_class],
                target_class_index, class_prob,
                single_attribute_differences)

            errors.append(error)

            # TODO: fix this, it does not correspond exactly to the paper, it should return the past
            #       error I believe
            # If we have reached the minimum or we are stuck in a local minimum
            if (error < ERROR_THRESHOLD) or ((abs(error) - abs(
                    old_error)) > ERROR_DIFFERENCE_THRESHOLD and not first_iteration):
                break
            else:
                first_iteration = False
                old_error = error

        if os.path.exists(join(DEFAULT_DIR, self.unique_filename)):
            shutil.rmtree(join(DEFAULT_DIR, self.unique_filename))

        print("explain_instance errors:", ", ".join([f"{err:.3E}" for err in errors]))

        xp = {'XPLAIN_explainer_o': self, 'diff_single': single_attribute_differences,
              'map_difference': deepcopy(difference_map), 'k': k, 'error': error,
              'instance': self.train_dataset.inverse_transform_instance(encoded_instance),
              'target_class': decoded_target_class,
              'errors': errors,
              'instance_class_index': target_class_index, 'prob': self.clf.predict_proba(
                encoded_instance_x)[0][target_class_index]}

        return xp

    def compute_lace_step(self,
                          encoded_instance,
                          k, target_class_frequency,
                          target_class_index, class_prob, single_attribute_differences):
        # Generate the neighborhood of the instance, classify it, and return the rules created by L3
        l3clf = L3Classifier(min_sup=self.min_sup)
        rules = create_locality_and_get_rules(self.train_dataset, self.nbrs, encoded_instance, k,
                                              self.unique_filename, self.clf, l3clf)

        # For each rule, calculate the prediction difference for the its attributes
        difference_map = {}
        for rule in rules:
            rule_key = ",".join(map(str, rule))
            difference_map[rule_key] = compute_prediction_difference_subset(
                self.train_dataset, encoded_instance, rule,
                self.clf, target_class_index)

        # Compute the approximation error
        _, error, _ = compute_approximation_error(
            target_class_frequency,
            class_prob,
            single_attribute_differences,
            rules,
            difference_map)

        return difference_map, error

    def getGlobalExplanationRules(self):
        # noinspection PyUnresolvedReferences
        from src.global_explanation import GlobalExplanation
        global_expl = GlobalExplanation(self)
        global_expl = global_expl.getGlobalExplanation()
        return global_expl


def create_locality_and_get_rules(training_dataset: Dataset, nbrs,
                                  encoded_instance: pd.Series, k: int,
                                  unique_filename: str, clf, l3clf):
    cc = training_dataset.class_column_name()
    instance_x = encoded_instance[:-1].to_numpy()

    nearest_neighbors_ixs = nbrs.kneighbors([instance_x], k, return_distance=False)[0]

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

    p = DEFAULT_DIR + unique_filename
    if not os.path.exists(p):
        os.makedirs(p)

    l3clf.fit(classified_instances_dataset.X_decoded(),
              classified_instances_dataset.Y_decoded(),
              column_names=classified_instances_dataset.X_decoded().columns.to_list())

    # Drop rules which use values not in the decoded instance
    decoded_instance = training_dataset.inverse_transform_instance(encoded_instance)
    encoded_rules = l3clf.lvl1_rules_

    def decode_rule(r_, clf_):
        r_class = clf_._class_dict[r_.class_id]
        r_attr_ixs_and_values = sorted([clf_._item_id_to_item[i] for i in r_.item_ids])
        r_attrs_and_values = [(clf_._column_id_to_name[c], v) for c, v in r_attr_ixs_and_values]
        return {'body': r_attrs_and_values, 'class': r_class}

    rules = []

    # Perform matching: remove all rules thta use an attibute value not present in the instance to
    # explain
    # NOTE: this matching does not consider the class. It considers rules regardless of their class
    #       prediction

    # For each rule
    for r in encoded_rules:
        # For each of its attributes and values
        for a, v in decode_rule(r, l3clf)['body']:
            # If rule uses an attribute's value different from the instance's
            if decoded_instance[a] != v:
                # Exit the inner loop, not entering the else clause, therefore not adding the rule
                break
        # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
        else:
            # If the inner loop has completed normally without break-ing, then all of the rule's
            # attribute values are in the instance as well, so we will use this rule

            # Get the instance attribute index from the rule's item_ids
            if decode_rule(r, l3clf)['class'] == decoded_instance.iloc[-1]:
                di = decoded_instance.index
                rules.append(
                    list(sorted([di.get_loc(a) + 1 for a, v in decode_rule(r, l3clf)['body']])))

    # # Print the instance
    # print('Instance:', ", ".join([f"{k}={v}" for k, v in
    #                               training_dataset.inverse_transform_instance(
    #                                   encoded_instance).to_dict().items()]))
    # print()
    # # Print l3wrapper's raw rules
    # for r in encoded_rules:
    #     print(r.item_ids, ", ".join([f"{k}={v}"
    #                                  for k, v in decode_rule(r, l3clf)['body']]), '->',
    #           decode_rule(r, l3clf)['class'])
    # # Print matched rules
    # print()
    # print('Rules:', rules)
    # print()

    # Get the union rule
    union_rule = list(sorted(set(itertools.chain.from_iterable(rules))))
    if union_rule not in rules and len(union_rule) > 0:
        rules.append(union_rule)

    rules = sorted(list(rules))

    return rules


def compute_perturbed_difference(item, clf, encoded_instance,
                                 instance_class_index,
                                 rule_attributes, training_dataset):
    (attribute_set, occurrences) = item

    perturbed_instance = deepcopy(encoded_instance)
    for i in range(len(rule_attributes)):
        perturbed_instance[rule_attributes[i]] = attribute_set[i]

    prob = clf.predict_proba(perturbed_instance[:-1].to_numpy().reshape(1, -1))[0][
        instance_class_index]

    # Compute the prediction difference using the weighted average of the
    # probability over the frequency of this attribute set in the
    # dataset
    difference = prob * occurrences / len(training_dataset)
    return difference


def compute_prediction_difference_single(encoded_instance, clf, class_prob, target_class_index,
                                         training_dataset):
    attribute_pred_difference = [0] * len(training_dataset.attributes())

    # For each attribute of the instance
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
            class_prob = clf.predict_proba(perturbed_encoded_instance_x.reshape(1, -1))[0][
                target_class_index]

            # Update the attribute difference weighting the prediction by the value frequency
            weight = attr_occurrences[attr_val] / len(training_dataset)
            difference = class_prob * weight
            attribute_pred_difference[attr_ix] += difference

    for i in range(len(attribute_pred_difference)):
        attribute_pred_difference[i] = class_prob - attribute_pred_difference[i]

    return attribute_pred_difference


def compute_prediction_difference_subset(training_dataset: Dataset,
                                         encoded_instance: pd.Series,
                                         rule_body_indices,
                                         clf,
                                         instance_class_index):
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


def compute_approximation_error(class_frequency, class_prob, single_attribute_differences,
                                impo_rules_complete,
                                difference_map):
    PI = class_prob - class_frequency
    Sum_Deltas = sum(single_attribute_differences)
    # UPDATED_EP
    if len(impo_rules_complete) > 0:
        impo_rules_completeC = ", ".join(map(str, list(max(impo_rules_complete, key=len))))
    else:
        impo_rules_completeC = ""

    approx_single_d = abs(PI - Sum_Deltas)
    approx_single_rel = approx_single_d / abs(PI)

    if impo_rules_completeC != "":
        if len(impo_rules_completeC.replace(" ", "").split(",")) > 1:
            Delta_impo_rules_completeC = difference_map[impo_rules_completeC.replace(" ", "")]
            PI_approx2 = Delta_impo_rules_completeC
            Sum_Deltas_not_in = 0.0
            # Sum of delta_i for each attribute not included
            for i_out_data in range(0, len(single_attribute_differences)):
                if str(i_out_data + 1) not in impo_rules_completeC.replace(" ", "").split(","):
                    Sum_Deltas_not_in = Sum_Deltas_not_in + single_attribute_differences[i_out_data]
        else:
            index = int(impo_rules_completeC.replace(" ", "").split(",")[0]) - 1
            PI_approx2 = single_attribute_differences[index]
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
