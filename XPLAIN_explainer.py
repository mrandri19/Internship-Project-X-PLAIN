# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import pickle
# noinspection PyUnresolvedReferences
import subprocess
# noinspection PyUnresolvedReferences
from copy import deepcopy

# noinspection PyUnresolvedReferences
import Orange
# noinspection PyUnresolvedReferences
import sklearn.neighbors

# noinspection PyUnresolvedReferences
from XPLAIN_explanation import XPLAIN_explanation
# noinspection PyUnresolvedReferences
from utils import gen_neighbors_info, \
    get_relevant_subset_from_local_rules, getClassifier_v2, import_datasets, import_dataset, \
    useExistingModel_v2, compute_prediction_difference_subset, \
    compute_prediction_difference_single, getStartKValueSimplified, \
    computeMappaClass_b, compute_error_approximation, createDir, convertOTable2Pandas, \
    get_KNN_threshold_max
# noinspection PyUnresolvedReferences
from global_explanation import *

ERROR_DIFFERENCE_THRESHOLD = 0.01
TEMPORARY_FOLDER_NAME = "tmp"
ERROR_THRESHOLD = 0.02


class XPLAIN_explainer:
    def __init__(self, dataset_name, classifier_name, classifier_parameter=None,
                 KneighborsUser=None, maxKNNUser=None, threshold_error=None,
                 use_existing_model=False, save_model=False,
                 random_explain_dataset=False):

        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.present = False

        # Temporary folder
        import uuid
        self.unique_filename = os.path.join(TEMPORARY_FOLDER_NAME,
                                            str(uuid.uuid4()))
        self.datanamepred = "./" + self.unique_filename + "/gen-k0.arff"
        should_exit = 0

        # The adult and compas dataset are already splitted in training and explain set.
        # The training set is balanced.
        self.explain_indices = []

        explain_dataset_indices = []
        if dataset_name == "datasets/adult_d.arff" \
                or dataset_name == "datasets/compas-scores-two-years_d.arff":
            self.training_dataset, self.explain_dataset, self.training_dataset_len, self.explain_indices = import_datasets(
                dataset_name, explain_dataset_indices, random_explain_dataset)
        else:
            self.training_dataset, self.explain_dataset, self.training_dataset_len, self.explain_indices = import_dataset(
                dataset_name, explain_dataset_indices, random_explain_dataset)

        self.K, _, self.max_K = get_KNN_threshold_max(KneighborsUser,
                                                      self.training_dataset_len,
                                                      threshold_error,
                                                      maxKNNUser)

        # If the user specifies to use an existing model, the model is used (if available).
        # Otherwise it is trained.
        if use_existing_model:
            # "Check if the model exist...
            self.classifier = useExistingModel_v2(classifier_name,
                                                  classifier_parameter,
                                                  dataset_name)
            if self.classifier:
                self.present = True
                # The model exists, we'll use it
            # The model does not exist, we'll train it")
        if use_existing_model is None or not self.present:
            self.classifier, should_exit, reason = getClassifier_v2(
                self.training_dataset, classifier_name, classifier_parameter,
                should_exit)

        if should_exit == 1:
            exit(-1)

        # Save the model only if required and it is not already saved.
        if save_model:
            # "Saving the model..."
            m = ""
            if classifier_parameter is not None:
                m = "-" + classifier_parameter
            createDir("./models")
            with open("./models/" + dataset_name + "-" + classifier_name + m,
                      "wb") as f:
                pickle.dump(self.classifier, f)

        self.map_names_class = {}
        num_i = 0
        for i in self.training_dataset.domain.class_var.values:
            self.map_names_class[num_i] = i
            num_i += 1
        self.labels = list(self.map_names_class.keys())

        self.dataset_name = dataset_name.split("/")[-1]

        self.NofClass = len(self.training_dataset.domain.class_var.values[:])

        # Compute the neighbors of the instanceId
        metric_knna = 'euclidean'
        self.NearestNeighborsAll = sklearn.neighbors.NearestNeighbors(
            n_neighbors=len(self.training_dataset), metric=metric_knna,
            algorithm='auto', metric_params=None).fit(self.training_dataset.X)
        self.mappa_single = {}

        self.firstInstance = 1

        self.starting_K = self.K

        self.mappa_class = computeMappaClass_b(self.training_dataset)
        self.count_inst = -1
        self.mispredictedInstances = None
        self.classes = list(self.map_names_class.values())

    def get_class_index(self, class_name):
        class_index = -1
        for i in self.training_dataset.domain.class_var.values:
            class_index += 1
            if i == class_name:
                return class_index

    def getMispredicted(self, mispred_class=False):
        self.mispredictedInstances = []
        count_inst = 0
        for n_ist in self.explain_indices:
            instanceI = Orange.data.Instance(self.explain_dataset.domain,
                                             self.explain_dataset[count_inst])
            c = self.classifier(instanceI, False)
            if instanceI.get_class() != self.map_names_class[c[0]]:
                if mispred_class is not False:
                    if instanceI.get_class() == mispred_class:
                        self.mispredictedInstances.append(n_ist)
                else:
                    self.mispredictedInstances.append(n_ist)
            count_inst = count_inst + 1
        return self.mispredictedInstances

    def explain_instance(self, instance, target_class):

        c = self.classifier(instance, False)
        target_class_index = self.get_class_index(target_class)

        self.starting_K = self.K
        # Problem with very small training dataset. The starting k is low, very few examples:
        # difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting k: proportional to the class frequence
        small_dataset_len = 150
        if self.training_dataset_len < small_dataset_len:
            self.starting_K = max(int(self.mappa_class[self.map_names_class[
                c[0]]] * self.training_dataset_len), self.starting_K)

        # Initialize k and error to be defined in case the for loop is not entered
        k = self.starting_K
        old_error = 10.0
        error = 1e9
        single_attribute_differences = {}
        pred = 0.0
        difference_map = {}

        first_iteration = True

        # Because across iterations only rules change we can cache both whole rules and instance
        # classifications
        cached_subset_differences = {}
        instance_predictions_cache = {}

        all_rule_body_indices = []

        # Euristically search for the best k to use to approximate the local model
        for k in range(self.starting_K, self.max_K, self.K):
            # Compute the prediction difference of single attributes only on the
            # first iteration
            if first_iteration:
                pred = self.classifier(instance, True)[0][target_class_index]
                single_attribute_differences = compute_prediction_difference_single(instance,
                                                                                    self.classifier,
                                                                                    target_class_index,
                                                                                    self.training_dataset)

            PI_rel2, difference_map, error, impo_rules_complete, importance_rules_lines, single_attribute_differences = self.compute_lace_step(
                cached_subset_differences, instance,
                instance_predictions_cache,
                k, all_rule_body_indices, target_class, target_class_index, pred,
                single_attribute_differences)

            # If we have reached the minimum or we are stuck in a local minimum
            if (error < ERROR_THRESHOLD) or ((abs(error) - abs(
                    old_error)) > ERROR_DIFFERENCE_THRESHOLD and not first_iteration):
                break
            else:
                first_iteration = False
                old_error = error
        instance_explanation = XPLAIN_explanation(self,
                                                  target_class,
                                                  instance,
                                                  single_attribute_differences,
                                                  k,
                                                  error,
                                                  difference_map)
        # Remove the temporary folder and dir
        import shutil
        if os.path.exists("./" + self.unique_filename):
            shutil.rmtree("./" + self.unique_filename)

        return instance_explanation

    def compute_lace_step(self, cached_subset_differences, instance,
                          instance_predictions_cache, k, old_input_ar, target_class,
                          target_class_index, pred, single_attribute_differences):
        print(f"compute_lace_step k={k}")

        gen_neighbors_info(self.training_dataset, self.NearestNeighborsAll, instance, k,
                           self.unique_filename, self.classifier)
        subprocess.call(['java', '-jar', 'AL3.jar', '-no-cv', '-t',
                         ('./' + self.unique_filename + '/Knnres.arff'), '-T',
                         ('./' + self.unique_filename + '/Filetest.arff'),
                         '-S', '1.0', '-C', '50.0', '-PN',
                         ("./" + self.unique_filename), '-SP', '10', '-NRUL',
                         '1'], stdout=subprocess.DEVNULL)
        with open("./" + self.unique_filename + "/impo_rules.txt",
                  "r") as myfile:
            importance_rules_lines = myfile.read().splitlines()
            # Remove rules which contain all attributes: we are not interested in a rule composed of
            # all the attributes values. By definition, its relevance is prob(y=c)-prob(c)
            importance_rules_lines = [rule_str for rule_str in importance_rules_lines if
                                      len(rule_str.split(",")) != len(instance.domain.attributes)]

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
                    self.training_dataset, instance, rule_body_indices,
                    self.classifier, target_class_index, instance_predictions_cache)

            difference_map_key = ",".join(map(str, rule_body_indices))
            difference_map[difference_map_key] = cached_subset_differences[
                subset_difference_cache_key]

        error_single, error, PI_rel2 = compute_error_approximation(self.mappa_class,
                                                                   pred,
                                                                   single_attribute_differences,
                                                                   impo_rules_complete,
                                                                   target_class,
                                                                   difference_map)
        old_input_ar += rule_bodies_indices

        return PI_rel2, difference_map, error, impo_rules_complete, importance_rules_lines, single_attribute_differences

    def showMispredictedTabularForm(self, mispred_class=False):
        sel = self.getMispredicted(mispred_class=mispred_class)
        sel_index = [self.explain_indices.index(i) for i in sel]
        return convertOTable2Pandas(self.explain_dataset, list(map(int, sel)),
                                    sel_index, self.classifier,
                                    self.map_names_class)

    # NEW_UPDATE
    def update_explain_instance(self, instance_explanation, rule_body_indices):
        target_class = instance_explanation.target_class
        instance = instance_explanation.instance
        target_class_index = instance_explanation.instance_class_index
        pred = self.classifier(instance, True)[0][target_class_index]

        difference_map = instance_explanation.map_difference

        # Because across iterations only rules change we can cache both whole rules and instance
        # classifications
        instance_predictions_cache = {}
        single_attribute_differences = instance_explanation.diff_single

        # Rule 1 element or already existing: no update needed
        if len(rule_body_indices) <= 1 or ','.join(map(str, rule_body_indices)) in difference_map:
            return instance_explanation

        PI_rel2, difference_map, error, impo_rules_complete = self.compute_prediction_difference_user_rule(
            rule_body_indices, instance,
            instance_predictions_cache,
            target_class, target_class_index, pred,
            single_attribute_differences, difference_map)

        instance_explanation = XPLAIN_explanation(self,
                                                  target_class,
                                                  instance,
                                                  single_attribute_differences,
                                                  instance_explanation.k,
                                                  error,
                                                  difference_map)

        return instance_explanation

    def compute_prediction_difference_user_rule(self, rule_body_indices, instance,
                                                instance_predictions_cache, target_class,
                                                target_class_index, pred,
                                                single_attribute_differences, difference_map):
        # Consider only rules with more than 1 attribute since we compute the differences
        # for single attribute changes already in compute_prediction_difference_single
        difference_map_key = ",".join(map(str, rule_body_indices))
        difference_map[difference_map_key] = compute_prediction_difference_subset(
            self.training_dataset, instance, rule_body_indices,
            self.classifier, target_class_index, instance_predictions_cache)

        impo_rules_complete = [list(map(int, e.split(","))) for e in list(difference_map.keys())]
        error_single, error, PI_rel2 = compute_error_approximation(self.mappa_class,
                                                                   pred,
                                                                   single_attribute_differences,
                                                                   impo_rules_complete,
                                                                   target_class,
                                                                   difference_map)

        return PI_rel2, difference_map, error, [impo_rules_complete]

    def getGlobalExplanationRules(self):
        global_expl = GlobalExplanation(self)
        global_expl = global_expl.getGlobalExplanation()
        return global_expl
