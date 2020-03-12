# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import pickle
# noinspection PyUnresolvedReferences
import subprocess
# noinspection PyUnresolvedReferences
from copy import deepcopy
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

# noinspection PyUnresolvedReferences
from src.XPLAIN_explanation import XPLAIN_explanation
# noinspection PyUnresolvedReferences
from src.dataset import Dataset
from src.global_explanation import *
# noinspection PyUnresolvedReferences
from src.utils import gen_neighbors_info, \
    get_relevant_subset_from_local_rules, get_classifier, import_datasets, import_dataset, \
    compute_prediction_difference_subset, \
    compute_prediction_difference_single, getStartKValueSimplified, \
    compute_class_frequency, compute_error_approximation, \
    get_KNN_threshold_max, DEFAULT_DIR, make_orange_instance, MT

ERROR_DIFFERENCE_THRESHOLD = 0.01
TEMPORARY_FOLDER_NAME = "tmp"
ERROR_THRESHOLD = 0.02


class XPLAIN_explainer:
    training_dataset: Dataset
    datanamepred: str
    unique_filename: str
    present: bool
    classifier_name: str
    dataset_name: str

    def __init__(self, dataset_name: str, classifier_name: str, classifier_parameter=None,
                 KneighborsUser=None, maxKNNUser=None, threshold_error=None,
                 random_explain_dataset=False):

        self.dataset_name = dataset_name
        self.classifier_name = classifier_name

        # Temporary folder
        import uuid
        self.unique_filename = os.path.join(TEMPORARY_FOLDER_NAME,
                                            str(uuid.uuid4()))
        self.datanamepred = DEFAULT_DIR + self.unique_filename + "/gen-k0.arff"

        # The adult and compas dataset are already splitted in training and explain set.
        # The training set is balanced.
        self.explain_indices = []

        explain_dataset_indices = []
        if dataset_name in [join(DEFAULT_DIR, "datasets/adult_d.arff"),
                            join(DEFAULT_DIR, "datasets/compas-scores-two-years_d.arff")]:
            self.training_dataset, self.explain_dataset, self.training_dataset_len, self.explain_indices = import_datasets(
                dataset_name, explain_dataset_indices, random_explain_dataset)
        else:
            self.training_dataset, self.explain_dataset, self.training_dataset_len, self.explain_indices = import_dataset(
                dataset_name, explain_dataset_indices, random_explain_dataset)

        self.K, _, self.max_K = get_KNN_threshold_max(KneighborsUser,
                                                      self.training_dataset_len,
                                                      threshold_error,
                                                      maxKNNUser)

        self.classifier = get_classifier(self.training_dataset, classifier_name,
                                         classifier_parameter)

        self.ix_to_class = {i: class_ for (i, class_) in
                            enumerate(self.training_dataset.class_values())}

        self.dataset_name = dataset_name.split("/")[-1]

        self.nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=len(self.training_dataset), metric='euclidean',
            algorithm='auto', metric_params=None).fit(
            self.training_dataset.X_numpy())

        self.starting_K = self.K

        self.class_frequencies = compute_class_frequency(self.training_dataset)
        self.mispredictedInstances = None

    def get_class_index(self, class_name):
        class_index = -1
        for i in self.training_dataset.class_values():
            class_index += 1
            if i == class_name:
                return class_index

    def explain_instance(self, decoded_instance: pd.Series, target_class) -> XPLAIN_explanation:
        orange_instance = make_orange_instance(self.explain_dataset, decoded_instance)
        target_class_index = self.get_class_index(target_class)

        encoded_instance = self.explain_dataset.transform_instance(decoded_instance)
        encoded_instance_x = encoded_instance[:-1].to_numpy()

        assert np.all(encoded_instance_x == orange_instance.x)

        self.starting_K = self.K
        # Problem with very small training dataset. The starting k is low, very few examples:
        # difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting k: proportional to the class frequence
        small_dataset_len = 150
        if self.training_dataset_len < small_dataset_len:
            pred_class = self.ix_to_class[
                self.classifier[MT].predict(encoded_instance_x.reshape(1, -1))[0]]
            self.starting_K = max(
                int(self.class_frequencies[pred_class] * self.training_dataset_len),
                self.starting_K)

        # Initialize k and error to be defined in case the for loop is not entered
        k = self.starting_K
        old_error = 10.0
        error = 1e9
        single_attribute_differences = {}
        orange_pred = 0.0
        difference_map = {}

        first_iteration = True

        # Because across iterations only rules change we can cache both whole rules and instance
        # classifications
        cached_subset_differences = {}
        instance_predictions_cache = {}

        all_rule_body_indices = []

        errors = []

        # Euristically search for the best k to use to approximate the local model
        for k in range(self.starting_K, self.max_K, self.K):
            # Compute the prediction difference of single attributes only on the
            # first iteration
            if first_iteration:
                orange_pred = \
                    self.classifier[MT].predict_proba(encoded_instance_x.reshape(1, -1))[0][
                        target_class_index]
                single_attribute_differences = compute_prediction_difference_single(
                    encoded_instance,
                    self.classifier,
                    target_class_index,
                    self.training_dataset)

            PI_rel2, difference_map, error, impo_rules_complete, importance_rules_lines, single_attribute_differences = self.compute_lace_step(
                cached_subset_differences, orange_instance, encoded_instance,
                instance_predictions_cache,
                k, all_rule_body_indices, target_class, target_class_index, orange_pred,
                single_attribute_differences)

            # TODO(Andrea): investigate this id=4943, <=50K
            # compute_lace_step k=118
            # compute_lace_step k=236
            # compute_lace_step k=354
            # compute_lace_step k=472
            # compute_lace_step k=590
            # explain_instance errors:
            # [0.49673809098087984,
            # 0.49673809098087984,
            # 0.49673809098087984,
            # 0.49673809098087984,
            # 0.49673809098087984]
            errors.append(error)

            # If we have reached the minimum or we are stuck in a local minimum
            if (error < ERROR_THRESHOLD) or ((abs(error) - abs(
                    old_error)) > ERROR_DIFFERENCE_THRESHOLD and not first_iteration):
                break
            else:
                first_iteration = False
                old_error = error

        instance_explanation = XPLAIN_explanation(self,
                                                  target_class,
                                                  orange_instance,
                                                  single_attribute_differences,
                                                  k,
                                                  error,
                                                  difference_map)
        # # Remove the temporary folder and dir
        import shutil
        if os.path.exists(join(DEFAULT_DIR, self.unique_filename)):
            shutil.rmtree(join(DEFAULT_DIR, self.unique_filename))

        print("explain_instance errors:", errors)

        return instance_explanation

    def compute_lace_step(self, cached_subset_differences, orange_instance,
                          encoded_instance,
                          instance_predictions_cache, k, old_input_ar, target_class,
                          target_class_index, pred, single_attribute_differences):
        print(f"compute_lace_step k={k}")

        gen_neighbors_info(self.training_dataset, self.nbrs, encoded_instance, k,
                           self.unique_filename, self.classifier)
        subprocess.call(['java', '-jar', DEFAULT_DIR + 'AL3.jar', '-no-cv', '-t',
                         (DEFAULT_DIR + self.unique_filename + '/Knnres.arff'), '-T',
                         (DEFAULT_DIR + self.unique_filename + '/Filetest.arff'),
                         '-S', '1.0', '-C', '50.0', '-PN',
                         (DEFAULT_DIR + self.unique_filename), '-SP', '10', '-NRUL',
                         '1'], stdout=subprocess.DEVNULL)
        with open(DEFAULT_DIR + self.unique_filename + "/impo_rules.txt",
                  "r") as myfile:
            importance_rules_lines = myfile.read().splitlines()
            # Remove rules which contain all attributes: we are not interested in a rule composed of
            # all the attributes values. By definition, its relevance is prob(y=c)-prob(c)
            importance_rules_lines = [rule_str for rule_str in importance_rules_lines if
                                      len(rule_str.split(",")) != (len(encoded_instance) - 1)]

        rule_bodies_indices, n_input_ar, new_input_ar, old_ar_set = \
            get_relevant_subset_from_local_rules(
                importance_rules_lines, old_input_ar)
        impo_rules_complete = deepcopy(rule_bodies_indices)

        # Cache the subset calculation for repeated rule subsets.
        difference_map = {}
        for rule_body_indices in rule_bodies_indices:
            # Consider only rules with more than 1 attribute since we compute the differences
            # for single attribute changes already in compute_prediction_difference_single
            if len(rule_body_indices) == 1:
                # Update Eliana - To output also rule of one element
                difference_map[str(rule_body_indices[0])] = single_attribute_differences[
                    rule_body_indices[0] - 1]
                continue
            if len(rule_body_indices) < 1:
                continue

            subset_difference_cache_key = tuple(rule_body_indices)
            if subset_difference_cache_key not in cached_subset_differences:
                cached_subset_differences[
                    subset_difference_cache_key] = compute_prediction_difference_subset(
                    self.training_dataset, orange_instance, rule_body_indices,
                    self.classifier, target_class_index, instance_predictions_cache)

            difference_map_key = ",".join(map(str, rule_body_indices))
            difference_map[difference_map_key] = cached_subset_differences[
                subset_difference_cache_key]

        error_single, error, PI_rel2 = compute_error_approximation(self.class_frequencies,
                                                                   pred,
                                                                   single_attribute_differences,
                                                                   impo_rules_complete,
                                                                   target_class,
                                                                   difference_map)
        old_input_ar += rule_bodies_indices

        return PI_rel2, difference_map, error, impo_rules_complete, importance_rules_lines, single_attribute_differences

    def getGlobalExplanationRules(self):
        global_expl = GlobalExplanation(self)
        global_expl = global_expl.getGlobalExplanation()
        return global_expl
