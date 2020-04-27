"""The module provides the XPLAIN_explainer class, which is used to perform the
$name analysis on a dataset.

"""
import itertools
from collections import Counter
from copy import deepcopy

import pandas as pd
import sklearn.neighbors
from l3wrapper.l3wrapper import L3Classifier

from src.dataset import Dataset

ERROR_DIFFERENCE_THRESHOLD = 0.01
ERROR_THRESHOLD = 0.02
MINIMUM_SUPPORT = 0.01
SMALL_DATASET_LEN = 150


class XPLAIN_explainer:
    """The XPLAIN_explainer class, through the `explain_instance` method allows
    to obtain a rule-based model-agnostic local explanation for an instance.

    Parameters
    ----------
    clf : sklearn classifier
        Any sklearn-like classifier can be passed. It must have the methods
        `predict` and `predict_proba`.
    train_dataset : Dataset
        The dataset from which the locality of the explained instance is created.
    min_sup : float
        L^3 Classifier's Minimum support parameter.
    """

    def __init__(self, clf, train_dataset, min_sup=MINIMUM_SUPPORT):
        self.clf = clf

        self.train_dataset = train_dataset

        self.K, self.max_K = _get_KNN_threshold_max(len(self.train_dataset))
        self.starting_K = self.K

        self.nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=len(self.train_dataset), metric='euclidean',
            algorithm='auto', metric_params=None).fit(
            self.train_dataset.X_numpy())

        self.min_sup = min_sup

        self.decoded_class_frequencies = self.train_dataset.Y_decoded().value_counts(normalize=True)

    def explain_instance(self, encoded_instance: pd.Series, decoded_target_class):
        """
        Explain the classifer's prediction on the instance with respect to a
        target class.

        Parameters
        ----------

        encoded_instance :
            The instance whose prediction needs to be explained. It may come
            from a :class:`~src.dataset.Dataset`. In that case, the `encoded`
            form an instance must be used i.e. :code:`my_dataset.X()[42]`, as
            opposed to the `decoded` form i.e. :code:`my_dataset.X_decoded()[42]`

        decoded_target_class : str
            The name of the class for which the prediction is explained. It may
            come from a :class:`~src.dataset.Dataset`. In that case, the
            `decoded` form must be used i.e. :code:`my_dataset.class_values()[3]`.

        Returns
        -------
        explanation : dict
            The explanation

            ::

                {
                    'XPLAIN_explainer_o': <src.XPLAIN_explainer.XPLAIN_explainer object at 0x7fde70c7a828>,
                    'diff_single': [0.11024691358024685, 0.02308641975308645, 0.19728395061728388, 0.31407407407407417, 0.004938271604938316, 0.006913580246913575, 0.0, 0.07111111111111112, 0.00864197530864197, 0.03358024691358019, 0.0007407407407408195, 0.0, 0.005185185185185182, 0.0, 0.0, 0.0],
                    'map_difference': {'1,2,3,4,8,9,10,11': 0.5839506172839506},
                    'k': 33,
                    'error': 0.00864197530864197,
                    'instance':
                        hair             1
                        feathers         0
                        eggs             0
                        milk             1
                        airborne         0
                        aquatic          0
                        predator         0
                        toothed          1
                        backbone         1
                        breathes         1
                        venomous         0
                        fins             0
                        legs             4
                        tail             1
                        domestic         0
                        catsize          1
                        type        mammal
                        dtype: object,
                    'target_class': 'mammal',
                    'errors': [0.00864197530864197],
                    'instance_class_index': 5,
                    'prob': 1.0
                }
            Let's examine each field:
        """
        target_class_index = self.train_dataset.class_values().index(decoded_target_class)

        encoded_instance_x = encoded_instance[:-1].to_numpy().reshape(1, -1)

        # Problem with very small training dataset. The starting k is low, very few examples:
        # difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting k: proportional to the class frequence
        training_dataset_len = len(self.train_dataset)
        if training_dataset_len < SMALL_DATASET_LEN:
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
        single_attribute_differences = _compute_prediction_difference_single(
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
            error = self._compute_lace_step(
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

        print("explain_instance errors:", ", ".join([f"{err:.3E}" for err in errors]))

        xp = {'XPLAIN_explainer_o': self, 'diff_single': single_attribute_differences,
              'map_difference': deepcopy(difference_map), 'k': k, 'error': error,
              'instance': self.train_dataset.inverse_transform_instance(encoded_instance),
              'target_class': decoded_target_class,
              'errors': errors,
              'instance_class_index': target_class_index, 'prob': self.clf.predict_proba(
                encoded_instance_x)[0][target_class_index]}

        return xp

    def _compute_lace_step(self,
                           encoded_instance,
                           k, target_class_frequency,
                           target_class_index, class_prob, single_attribute_differences):
        # Generate the neighborhood of the instance, classify it, and return the rules created by L3
        l3clf = L3Classifier(min_sup=self.min_sup)
        rules = _create_locality_and_get_rules(self.train_dataset, self.nbrs, encoded_instance, k,
                                               self.clf, l3clf)

        # For each rule, calculate the prediction difference for the its attributes
        difference_map = {}
        for rule in rules:
            rule_key = ",".join(map(str, rule))
            difference_map[rule_key] = _compute_prediction_difference_subset(
                self.train_dataset, encoded_instance, rule,
                self.clf, target_class_index)

        # Compute the approximation error
        _, error, _ = _compute_approximation_error(
            target_class_frequency,
            class_prob,
            single_attribute_differences,
            rules,
            difference_map)

        return difference_map, error

    def _getGlobalExplanationRules(self):
        # noinspection PyUnresolvedReferences
        from src.global_explanation import GlobalExplanation
        global_expl = GlobalExplanation(self)
        global_expl = global_expl.getGlobalExplanation()
        return global_expl


def _create_locality_and_get_rules(training_dataset: Dataset, nbrs,
                                   encoded_instance: pd.Series, k: int,
                                   clf, l3clf):
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

    # Perform matching: remove all rules that use an attibute value not present in the instance to
    # explain

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

    # Get the union rule
    union_rule = list(sorted(set(itertools.chain.from_iterable(rules))))
    if union_rule not in rules and len(union_rule) > 0:
        rules.append(union_rule)

    rules = sorted(list(rules))

    return rules


def _compute_perturbed_difference(item, clf, encoded_instance,
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


def _compute_prediction_difference_single(encoded_instance, clf, class_prob, target_class_index,
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


def _compute_prediction_difference_subset(training_dataset: Dataset,
                                          encoded_instance: pd.Series,
                                          rule_body_indices,
                                          clf,
                                          instance_class_index):
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
        _compute_perturbed_difference(item, clf, encoded_instance,
                                      instance_class_index,
                                      rule_attributes, training_dataset) for
        item in
        attribute_sets_occurrences.items()]

    prediction_difference = sum(differences)

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = clf.predict_proba(encoded_instance_x.reshape(1, -1))[0][instance_class_index]
    prediction_differences = p - prediction_difference

    return prediction_differences


def _compute_approximation_error(class_frequency, class_prob, single_attribute_differences,
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


def _get_KNN_threshold_max(len_dataset):
    import math
    k = int(round(math.sqrt(len_dataset)))

    if len_dataset < 150:
        max_n = len_dataset
    elif len_dataset < 1000:
        max_n = int(len_dataset / 2)
    elif len_dataset < 10000:
        max_n = int(len_dataset / 10)
    else:
        max_n = int(len_dataset * 5 / 100)

    return k, max_n
