# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import subprocess

# noinspection PyUnresolvedReferences
import sklearn.neighbors

from XPLAIN_explanation_class import XPLAIN_explanation
# noinspection PyUnresolvedReferences
from XPLAIN_utils.LACE_utils.LACE_utils2 import getStartKValueSimplified, \
    computeMappaClass_b, computeApproxError
# noinspection PyUnresolvedReferences
from XPLAIN_utils.LACE_utils.LACE_utils3 import genNeighborsInfo, \
    get_relevant_subset_from_local_rules, getClassifier_v2
from XPLAIN_utils.LACE_utils.LACE_utils4 import *

ERROR_DIFFERENCE_THRESHOLD = 0.01
TEMPORARY_FOLDER_NAME = "tmp"
ERROR_THRESHOLD = 0.02


class XPLAIN_explainer:
    def __init__(self, dataset_name, classifier_name, classifier_parameter=None,
                 KneighborsUser=None, maxKNNUser=None, threshold_error=None,
                 use_existing_model=False, save_model=False,
                 train_explain_set=False):

        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.present = False
        self.evaluate_explanation = False

        # Temporary folder
        import uuid
        self.unique_filename = os.path.join(TEMPORARY_FOLDER_NAME,
                                            str(uuid.uuid4()))
        should_exit = 0

        # The adult and compas dataset are already splitted in training and explain set.
        # The training set is balanced.
        self.instance_indices = []
        if dataset_name == "datasets/adult_d.arff" \
                or dataset_name == "datasets/compas-scores-two-years_d.arff":
            self.training_dataset, self.explain_dataset, self.len_dataset, self.instance_indices = \
                import_datasets(
                    dataset_name, self.instance_indices, train_explain_set, False)
        else:
            self.training_dataset, self.explain_dataset, self.len_dataset, self.instance_indices = \
                import_dataset(
                    dataset_name, self.instance_indices, train_explain_set)

        self.K, _, self.max_K = get_KNN_threshold_max(KneighborsUser,
                                                      self.len_dataset,
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
        if use_existing_model is None or self.present == False:
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
        self.map_instance_apprE = {}
        self.map_instance_NofKNN = {}
        self.map_instance_1_apprE = {}
        self.map_instance_diff_approxFirst = {}
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
        for n_ist in self.instance_indices:
            instanceI = Orange.data.Instance(self.explain_dataset.domain,
                                             self.explain_dataset[count_inst])
            c = self.classifier(instanceI, False)
            if instanceI.get_class() != self.map_names_class[c[0]]:
                if mispred_class != False:
                    if instanceI.get_class() == mispred_class:
                        self.mispredictedInstances.append(n_ist)
                else:
                    self.mispredictedInstances.append(n_ist)
            count_inst = count_inst + 1
        return self.mispredictedInstances

    def interactiveTargetClassComparison(self, instID):
        from ipywidgets import HBox, VBox
        classes = ["predicted", "trueLabel"] + self.classes[:]
        w1 = widgets.Dropdown(
            options=classes,
            description='1º',
            value="predicted",
            disabled=False
        )
        w2 = widgets.Dropdown(
            options=classes,
            description='2º',
            value="trueLabel",
            disabled=False
        )
        hClasses = VBox([w1, w2])
        l = widgets.Label(value='Select target classes:')
        display(l)
        display(hClasses)

        def clearAndShow(btNewObj):
            clear_output()
            display(l)
            display(hClasses)
            display(h)

        def getExplainInteractiveButton(btn_object):
            e1, e2 = self.getExplanationComparison(instID, w1.value, w2.value)

        btnTargetC = widgets.Button(description='Compute')
        btnTargetC.on_click(getExplainInteractiveButton)
        btnNewSel = widgets.Button(description='Clear')
        btnNewSel.on_click(clearAndShow)
        h = HBox([btnTargetC, btnNewSel])
        display(h)

    def getMispredictedTrueLabelComparison(self, instID):
        e1, e2 = self.getExplanationComparison(instID, "predicted", "trueLabel")

    def comparePerturbed(self, Sn_inst, Sn_inst1, inst1, indexes=[]):

        fig2 = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig2.add_subplot(1, 2, 1)
        explanation_1, ax1 = self.getExplanation_i_axis(ax1, Sn_inst)
        ax2 = fig2.add_subplot(1, 2, 2)
        explanation_2, ax2 = self.getExplanationPerturbed_i_axis(ax2, Sn_inst1,
                                                                 inst1)

        for i in indexes:
            ax2.get_yticklabels()[i].set_color("red")
        plt.tight_layout()
        plt.show()
        return explanation_1, explanation_2

    def getExplanationComparison(self, Sn_inst, targetClass1,
                                 targetClass2=None):

        if targetClass1 == targetClass2:
            print("Same target class")
            return self.explain_instance(Sn_inst, targetClass1), None

        if targetClass1 == "predicted" and targetClass2 == None:
            print("Predicted class")
            return self.explain_instance(Sn_inst), None

        predicted, true = self.getPredictedandTrueClassById(Sn_inst)

        if targetClass1 == None:
            targetClass1 = "predicted"
        if targetClass2 == None:
            targetClass2 = "predicted"

        if targetClass1 == "predicted" or targetClass2 == "predicted":
            if predicted == targetClass1 or predicted == targetClass2:
                print("Predicted class = user target class ")
                return self.explain_instance(Sn_inst), None
            if targetClass1 == "trueLabel" or targetClass2 == "trueLabel":
                if true == predicted:
                    print("True class = predicted class ")
                    return self.explain_instance(Sn_inst), None
        if targetClass1 == "trueLabel" or targetClass2 == "trueLabel":
            if true == targetClass1 or true == targetClass2:
                print("True class = user target class ")
                return self.explain_instance(Sn_inst), None

        fig2 = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig2.add_subplot(1, 2, 1)
        explanation_1, ax1 = self.getExplanation_i_axis(ax1, Sn_inst,
                                                        targetClass1)
        ax2 = fig2.add_subplot(1, 2, 2)
        explanation_2, ax2 = self.getExplanation_i_axis(ax2, Sn_inst,
                                                        targetClass2)
        plt.tight_layout()
        plt.show()
        return explanation_1, explanation_2

    def getInstanceById(self, Sn_inst):
        count_inst = self.instance_indices.index(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        return instTmp2

    def getPredictedandTrueClassById(self, Sn_inst):
        i = self.getInstanceById(Sn_inst)
        c = self.classifier(i, False)
        return self.map_names_class[c[0]], str(i.get_class())

    def getPredictedandTrueClassByInstance(self, i):
        c = self.classifier(i, False)
        return self.map_names_class[c[0]], str(i.get_class())

    def getExplanationPerturbed_i_axis(self, axi, Sn_inst, instTmp2,
                                       targetClass=None):

        oldinputAr = []
        oldMappa = {}
        count_inst = -1

        old_impo_rules = []
        firstKNN = True
        oldError = 10.0
        # count_inst=self.n_insts.index(Sn_inst)

        # n_inst2=int(Sn_inst)
        # instTmp2 = Orange.data.Instance(self.explain_dataset.domain, self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)

        if targetClass == None:
            # default
            targetClass = self.map_names_class[c[0]]
        elif targetClass == "trueLabel":
            targetClass = str(instTmp2.get_class())
        else:
            targetClass = str(targetClass)
        indexI = self.get_class_index(targetClass)

        self.starting_K = self.K
        # Problem with very small training dataset. The starting KNN is low, very few examples: difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting K: proportional to the class frequence
        small_dataset_len = 150
        if self.len_dataset < small_dataset_len:
            self.starting_K = max(int(self.mappa_class[self.map_names_class[
                c[0]]] * self.len_dataset), self.starting_K)

        plot = False
        for NofKNN in range(self.starting_K, self.max_K, self.K):
            # DO TO MULTIPLE
            instTmp = deepcopy(instTmp2)
            n_inst = Sn_inst
            instT = deepcopy(instTmp2)

            genNeighborsInfo(self.training_dataset, self.NearestNeighborsAll,
                             instT, 0, NofKNN, self.unique_filename,
                             self.classifier)

            # Call L3
            subprocess.call(['java', '-jar', 'AL3.jar', '-no-cv', '-t',
                             './' + self.unique_filename + '/Knnres.arff', '-T',
                             './' + self.unique_filename + '/Filetest.arff',
                             '-S', '1.0', '-C', '50.0', '-PN',
                             "./" + self.unique_filename, '-SP', '10', '-NRUL',
                             '1'])

            self.datanamepred = "./" + self.unique_filename + "/gen-k0.arff"
            with open("./" + self.unique_filename + "/impo_rules.txt",
                      "r") as myfile:
                impo_rules = myfile.read().splitlines()

            # The local model is not changed
            if set(impo_rules) == set(old_impo_rules) and firstKNN == False:
                continue

            old_impo_rules = impo_rules[:]

            impo_rules_N = []

            reduceOverfitting = False
            for impo_r in impo_rules:
                # Not interested in a rule composed of all the attributes values. By definition, its relevance is prob(y=c)-prob(c)
                if len(impo_r.split(",")) != len(instTmp.domain.attributes):
                    impo_rules_N.append(impo_r)

            impo_rules = impo_rules_N[:]

            inputAr, nInputAr, newInputAr, oldAr_set = get_relevant_subset_from_local_rules(
                impo_rules, oldinputAr)

            impo_rules_complete = deepcopy(inputAr)

            # Compute probability of Single attribute or Set of Attributes
            firstInstance = 0
            mappaNew = {}
            mappa = oldMappa.copy()
            mappa.update(mappaNew)

            if firstKNN:
                c1 = self.classifier(instT, True)[0]
                pred = c1[indexI]
                pred_str = str(round(c1[indexI], 2))
                out_data = compute_prediction_difference_single(instT,
                                                                self.classifier,
                                                                indexI,
                                                                self.training_dataset)

            map_difference = {}
            map_difference = computePredictionDifferenceSubsetRandomOnlyExisting(
                self.training_dataset, instT, inputAr, targetClass,
                self.classifier, indexI, map_difference)

            # Definition of approximation error. How we approximate the "true explanation"?
            error_single, error, PI_rel2 = computeApproxError(self.mappa_class,
                                                              pred, out_data,
                                                              impo_rules_complete,
                                                              targetClass,
                                                              map_difference)

            minlen, minlenname, minvalue = getMinRelevantSet(instT,
                                                             impo_rules_complete,
                                                             map_difference)

            oldinputAr = inputAr + oldinputAr
            oldinputAr_set = set(map(tuple, oldinputAr))
            oldMappa.update(mappa)
            if firstKNN:
                self.map_instance_1_apprE[n_inst] = PI_rel2
                self.map_instance_diff_approxFirst[n_inst] = error
                self.map_instance_apprE[n_inst] = PI_rel2
                self.map_instance_NofKNN[n_inst] = NofKNN

            # if (error)*100<threshold:
            threshold = 0.02
            if (error) < threshold:
                # final
                if not (self.evaluate_explanation):
                    plotTheInfo_axi(instT, out_data, impo_rules, n_inst,
                                    self.dataset_name, NofKNN, "f", minlenname,
                                    minvalue, targetClass, error, error_single,
                                    self.classifier_name, map_difference,
                                    impo_rules_complete, pred_str, axi)
                    explanation_i = XPLAIN_explanation(self, targetClass, instT,
                                                       out_data, impo_rules,
                                                       n_inst, NofKNN, error,
                                                       map_difference,
                                                       impo_rules_complete)
                self.map_instance_apprE[n_inst] = PI_rel2
                self.map_instance_NofKNN[n_inst] = NofKNN
                printImpoRuleInfo(n_inst, instT, NofKNN, out_data,
                                  map_difference, impo_rules_complete,
                                  impo_rules)
                plot = True
                break
            # local minimum
            elif (abs(error) - abs(oldError)) > 0.01 and firstKNN == False:
                # PLOT OLD ERROR AS BETTER
                if not (self.evaluate_explanation):
                    plotTheInfo_axi(instT, old_out_data, old_impo_rulesPlot,
                                    n_inst, self.dataset_name, oldNofKNN, "f",
                                    minlenname, minvalue, targetClass, oldError,
                                    error_single, self.classifier_name,
                                    old_map_difference, old_impo_rules_complete,
                                    pred_str, axi)
                    explanation_i = XPLAIN_explanation(self, targetClass, instT,
                                                       old_out_data,
                                                       old_impo_rulesPlot,
                                                       n_inst, oldNofKNN,
                                                       oldError,
                                                       old_map_difference,
                                                       old_impo_rules_complete)
                plot = True
                self.map_instance_apprE[n_inst] = PI_rel2_old
                self.map_instance_NofKNN[n_inst] = oldNofKNN
                printImpoRuleInfo(n_inst, instT, oldNofKNN, old_out_data,
                                  old_map_difference, old_impo_rules_complete,
                                  old_impo_rulesPlot)
                break
            else:
                firstKNN = False
                oldError = error
                oldNofKNN = NofKNN
                old_out_data = deepcopy(out_data)
                old_impo_rulesPlot = deepcopy(impo_rules)
                old_map_difference = deepcopy(map_difference)
                old_impo_rules_complete = deepcopy(impo_rules_complete)
                PI_rel2_old = PI_rel2

        # if NofKNN>=(self.maxN-startingK):
        if NofKNN >= (self.max_K) or plot == False:
            if (error) == (oldError):
                if not (self.evaluate_explanation):
                    plotTheInfo_axi(instT, old_out_data, old_impo_rulesPlot,
                                    n_inst, self.dataset_name, oldNofKNN, "f",
                                    minlenname, minvalue, targetClass, oldError,
                                    error_single, self.classifier_name,
                                    old_map_difference, old_impo_rules_complete,
                                    pred_str, axi)
                    explanation_i = XPLAIN_explanation(self, targetClass, instT,
                                                       old_out_data,
                                                       old_impo_rulesPlot,
                                                       n_inst, oldNofKNN,
                                                       oldError,
                                                       old_map_difference,
                                                       old_impo_rules_complete)
                self.map_instance_apprE[n_inst] = PI_rel2_old
                self.map_instance_NofKNN[n_inst] = oldNofKNN
                printImpoRuleInfo(n_inst, instT, oldNofKNN, old_out_data,
                                  old_map_difference, old_impo_rules_complete,
                                  old_impo_rulesPlot)

            else:
                if not (self.evaluate_explanation):
                    plotTheInfo_axi(instT, out_data, impo_rules, n_inst,
                                    self.dataset_name, NofKNN, "f", minlenname,
                                    minvalue, targetClass, error, error_single,
                                    self.classifier_name, map_difference,
                                    impo_rules_complete, pred_str, axi)
                    explanation_i = XPLAIN_explanation(self, targetClass, instT,
                                                       out_data, impo_rules,
                                                       n_inst, NofKNN, error,
                                                       map_difference,
                                                       impo_rules_complete)
                self.map_instance_apprE[n_inst] = PI_rel2
                self.map_instance_NofKNN[n_inst] = NofKNN
                printImpoRuleInfo(n_inst, instT, out_data, map_difference,
                                  impo_rules_complete, impo_rules)

        # Remove the temporary folder and dir
        import shutil
        if os.path.exists("./" + self.unique_filename):
            shutil.rmtree("./" + self.unique_filename)

        return explanation_i, axi

    def explain_instance(self, instance_index_str, target_class=None):
        oldinputAr = []
        oldMappa = {}

        old_impo_rules = []
        oldError = 10.0
        instance_index_in_instance_indices = self.instance_indices.index(
            instance_index_str)

        instance = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[
                                            instance_index_in_instance_indices])
        c = self.classifier(instance, False)

        if target_class is None:
            target_class = self.map_names_class[c[0]]
        elif target_class == "predicted":
            target_class = self.map_names_class[c[0]]
        elif target_class == "trueLabel":
            target_class = str(instance.get_class())
        else:
            target_class = str(target_class)
        target_class_index = self.get_class_index(target_class)

        instance_explanation = None

        self.starting_K = self.K
        # Problem with very small training dataset. The starting KNN is low, very few examples: difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting k: proportional to the class frequence
        small_dataset_len = 150
        if self.len_dataset < small_dataset_len:
            self.starting_K = max(int(self.mappa_class[self.map_names_class[
                c[0]]] * self.len_dataset), self.starting_K)

        # Initialize k and error to be defined in case the for loop is not entered
        k = 0
        error = 1e9

        first_iteration = True

        # TODO(andrea): verify with Eliana this assumptions, the snapshot test data works
        # Because across iterations only rules change we can cache both whole rules and instance
        # classifications
        cached_subset_differences = {}
        instance_predictions_cache = {}

        # Euristically search for the best k to use to approximate the local model
        for k in range(self.starting_K, self.max_K, self.K):
            print(f"Trying k={k}")
            instance_index = int(instance_index_str)

            genNeighborsInfo(self.training_dataset, self.NearestNeighborsAll,
                             self.explain_dataset[
                                 instance_index_in_instance_indices],
                             instance_index, k,
                             self.unique_filename, self.classifier)

            # Call L3
            subprocess.call(['java', '-jar', 'AL3.jar', '-no-cv', '-t',
                             './' + self.unique_filename + '/Knnres.arff', '-T',
                             './' + self.unique_filename + '/Filetest.arff',
                             '-S', '1.0', '-C', '50.0', '-PN',
                             "./" + self.unique_filename, '-SP', '10', '-NRUL',
                             '1'], stdout=subprocess.DEVNULL)

            self.datanamepred = "./" + self.unique_filename + "/gen-k0.arff"
            with open("./" + self.unique_filename + "/impo_rules.txt",
                      "r") as myfile:
                importance_rules_lines = myfile.read().splitlines()

            # The local model is not changed
            if set(importance_rules_lines) == set(
                    old_impo_rules) and not first_iteration:
                continue

            old_impo_rules = importance_rules_lines

            # Not interested in a rule composed of all the attributes values.
            # By definition, its relevance is prob(y=c)-prob(c)
            importance_rules_lines = [rule_str for rule_str in
                                      importance_rules_lines if
                                      len(rule_str.split(",")) != len(
                                          instance.domain.attributes)]

            rule_bodies_indices, nInputAr, newInputAr, oldAr_set = get_relevant_subset_from_local_rules(
                importance_rules_lines, oldinputAr)

            impo_rules_complete = deepcopy(rule_bodies_indices)

            # Compute probability of Single attribute or Set of Attributes
            mappa_new = {}
            mappa = oldMappa.copy()
            mappa.update(mappa_new)

            # Compute the prediction difference of single attributes only on the
            # first iteration
            if first_iteration:
                c1 = self.classifier(instance, True)[0]
                pred = c1[target_class_index]
                out_data = compute_prediction_difference_single(instance,
                                                                self.classifier,
                                                                target_class_index,
                                                                self.training_dataset)

            # Cache the subset calculation for repeated rule subsets. In this loop the only thing
            # that change are rules, right?
            difference_map = {}
            for rule_body_indices in rule_bodies_indices:
                # Consider only rules with more than 1 attribute since we compute the differences
                # for single attribute changes already in computePredictionDifferenceSinglever2
                if len(rule_body_indices) <= 1:
                    continue
                subset_difference_cache_key = tuple(rule_body_indices)
                if subset_difference_cache_key not in cached_subset_differences:
                    cached_subset_differences[
                        subset_difference_cache_key] = compute_prediction_difference_subset_only_existing(
                        self.training_dataset, instance, rule_body_indices,
                        self.classifier, target_class_index, instance_predictions_cache)

                difference_map_key = ",".join(map(str, rule_body_indices))
                difference_map[difference_map_key] = cached_subset_differences[
                    subset_difference_cache_key]

            # Definition of approximation error. How we approximate the "true explanation"?
            error_single, error, PI_rel2 = computeApproxError(self.mappa_class,
                                                              pred, out_data,
                                                              impo_rules_complete,
                                                              target_class,
                                                              difference_map)

            oldinputAr += rule_bodies_indices
            oldMappa.update(mappa)
            if first_iteration:
                self.map_instance_1_apprE[instance_index] = PI_rel2
                self.map_instance_diff_approxFirst[instance_index] = error
                self.map_instance_apprE[instance_index] = PI_rel2
                self.map_instance_NofKNN[instance_index] = k

            # If we have reached the minimum
            if error < ERROR_THRESHOLD:
                if not self.evaluate_explanation:
                    instance_explanation = XPLAIN_explanation(self,
                                                              target_class,
                                                              instance,
                                                              out_data,
                                                              importance_rules_lines,
                                                              instance_index, k,
                                                              error,
                                                              difference_map,
                                                              impo_rules_complete)
                self.map_instance_apprE[instance_index] = PI_rel2
                self.map_instance_NofKNN[instance_index] = k
                break
            # If we are stuck in a local minimum
            elif (abs(error) - abs(
                    oldError)) > ERROR_DIFFERENCE_THRESHOLD and not first_iteration:
                if not self.evaluate_explanation:
                    instance_explanation = XPLAIN_explanation(self,
                                                              target_class,
                                                              instance,
                                                              old_out_data,
                                                              old_impo_rulesPlot,
                                                              instance_index,
                                                              oldNofKNN,
                                                              oldError,
                                                              old_map_difference,
                                                              old_impo_rules_complete)
                self.map_instance_apprE[instance_index] = PI_rel2_old
                self.map_instance_NofKNN[instance_index] = oldNofKNN
                break
            # If we have neither reached the minimum nor in al local minimum
            else:
                first_iteration = False
                oldError = error
                oldNofKNN = k
                old_out_data = deepcopy(out_data)
                old_impo_rulesPlot = deepcopy(importance_rules_lines)
                old_map_difference = deepcopy(difference_map)
                old_impo_rules_complete = deepcopy(impo_rules_complete)
                PI_rel2_old = PI_rel2
        # If the for loop ended after having reached the maximum number of
        # iteration, i.e. the error did not reach the minimum.
        else:
            # If we are stuck, i.e. the error did not change between iterations
            if error == oldError:
                if not self.evaluate_explanation:
                    instance_explanation = XPLAIN_explanation(self,
                                                              target_class,
                                                              instance,
                                                              old_out_data,
                                                              old_impo_rulesPlot,
                                                              instance_index,
                                                              oldNofKNN,
                                                              oldError,
                                                              old_map_difference,
                                                              old_impo_rules_complete)
                self.map_instance_apprE[instance_index] = PI_rel2_old
                self.map_instance_NofKNN[instance_index] = oldNofKNN
            else:
                if not self.evaluate_explanation:
                    instance_explanation = XPLAIN_explanation(self,
                                                              target_class,
                                                              instance,
                                                              out_data,
                                                              importance_rules_lines,
                                                              instance_index, k,
                                                              error,
                                                              difference_map,
                                                              impo_rules_complete)
                self.map_instance_apprE[instance_index] = PI_rel2
                self.map_instance_NofKNN[instance_index] = k

        # Remove the temporary folder and dir
        import shutil
        if os.path.exists("./" + self.unique_filename):
            shutil.rmtree("./" + self.unique_filename)

        # instance_explanation cannot be None since it's set in all 4 cases:
        # 1. Minimum reached
        # 2. Local minimum reached
        # 3. Exceeded max_K and did not reduce the error in the last iteration
        # 4. Exceeded max_K and did reduce the error in the last iteration
        assert (instance_explanation is not None)

        return instance_explanation

    def visualizePoints(self, datapoints, Sn_inst=None, reductionMethod="mca"):
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn import decomposition
        if Sn_inst != None:
            count_inst = self.instance_indices.index(Sn_inst)
            n_inst = int(Sn_inst)
            instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                            self.explain_dataset[count_inst])
            c = self.classifier(instTmp2, False)
            labelledInstance = deepcopy(instTmp2)

        X = datapoints.X
        y = datapoints.Y

        if reductionMethod == "pca":
            pca = decomposition.PCA(n_components=3)
            pca.fit(X)
            X = pca.transform(X)
            if Sn_inst != None:
                istance_transformed = pca.transform([labelledInstance.x])

        elif reductionMethod == "mca":
            import pandas as pd
            import prince

            dataK = []
            for k in range(0, len(datapoints)):
                dataK.append(datapoints[k].list)

            columnsA = [i.name for i in datapoints.domain.variables]

            if datapoints.domain.metas != ():
                for i in range(0, len(datapoints.domain.metas)):
                    columnsA.append(datapoints.domain.metas[i].name)
            data = pd.DataFrame(data=dataK, columns=columnsA)

            columnsA = [i.name for i in datapoints.domain.attributes]
            Xa = data[columnsA]
            y = datapoints.Y

            mca = prince.MCA(n_components=3, n_iter=3, copy=True,
                             check_input=True, engine='auto', random_state=42)
            mca.fit(Xa)
            X = mca.transform(Xa)
            if Sn_inst != None:
                istance_transformed = mca.transform([[labelledInstance[i].value
                                                      for i in
                                                      labelledInstance.domain.attributes]])

        elif reductionMethod == "t-sne":
            from sklearn.manifold import TSNE
            if Sn_inst != None:
                XX = np.vstack([X, labelledInstance.x])
                label_istance = float(
                    max(list(self.map_names_class.keys())) + 1)
                yy = np.concatenate((y, np.array([label_istance])))
            else:
                XX = X
                yy = y
            tsne = TSNE(n_components=2, random_state=0)
            tsne.fit(XX)
            XX = tsne.fit_transform(XX)

        else:
            print("Reduction method available: pca, t-sne, selected",
                  reductionMethod)

        y_l = y.astype(int)
        labelMapNames = self.map_names_class.items()
        if Sn_inst != None:
            label_istance = float(max(list(self.map_names_class.keys())) + 1)
            instance_label_name = self.map_names_class[int(labelledInstance.y)]

        if reductionMethod == "pca" or reductionMethod == "mca":
            if Sn_inst != None:
                XX = np.vstack([X, istance_transformed])
                yy = np.concatenate((y, np.array([label_istance])))
            else:
                XX = X
                yy = y
            fig = plt.figure(figsize=(5.5, 3))
            ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            sc = ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=yy, cmap="Spectral",
                            edgecolor='k')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            label_values = list(np.unique(y_l))
            if Sn_inst != None:
                label_values.append(int(label_istance))
        else:
            fig, ax = plt.subplots()
            sc = ax.scatter(XX[:, 0], XX[:, 1], c=yy, cmap="tab10")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            label_values = list(np.unique(yy.astype(int)))

        colors = [sc.cmap(sc.norm(i)) for i in label_values]
        custom_lines = [plt.Line2D([], [], ls="", marker='.',
                                   mec='k', mfc=c, mew=.1, ms=20) for c in
                        colors]

        d2 = dict(labelMapNames)
        if Sn_inst != None:
            d2[int(label_istance)] = instance_label_name + "_i"
        labelMapNames_withInstance = d2.items()

        newdict = {k: dict(labelMapNames_withInstance)[k] for k in label_values}

        ax.legend(custom_lines, [lt[1] for lt in newdict.items()],
                  loc='center left', bbox_to_anchor=(1.0, .5))

        if reductionMethod == "t-sne":
            fig.tight_layout()

        plt.show()

    def showTrainingPoints(self, Sn_inst=None, reductionMethod="pca"):

        X = self.training_dataset.X
        y = self.training_dataset.Y
        self.visualizePoints(self.training_dataset, Sn_inst, reductionMethod)

    def showNNLocality(self, Sn_inst, reductionMethod="pca", training=False):
        count_inst = self.instance_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.len_dataset < small_dataset_len:
            self.starting_K = max(int(self.mappa_class[self.map_names_class[
                c[0]]] * self.len_dataset), self.K)
        if training == True:
            Kneighbors_data, removeToDo = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, removeToDo = genNeighborsInfo(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier, save=False)

        X = Kneighbors_data.X
        y = Kneighbors_data.Y
        self.visualizePoints(Kneighbors_data, Sn_inst, reductionMethod)

    def showNearestNeigh_type_2(self, Sn_inst, fig2, position,
                                reductionMethod="pca", training=False):

        from sklearn import decomposition

        count_inst = self.instance_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        # Plottarla con un colore diverso
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.len_dataset < small_dataset_len:
            self.starting_K = max(int(
                self.mappa_class[
                    self.map_names_class[c[0]]] * self.len_dataset),
                self.K)
        if training == True:
            Kneighbors_data, removeToDo = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, removeToDo = genNeighborsInfo(
                self.training_dataset,
                self.NearestNeighborsAll,
                self.explain_dataset[
                    count_inst], n_inst,
                self.starting_K,
                self.unique_filename,
                self.classifier,
                save=False)

        X = Kneighbors_data.X
        y = Kneighbors_data.Y
        labelledInstance = deepcopy(instTmp2)

        if reductionMethod == "pca":
            pca = decomposition.PCA(n_components=3)
            pca.fit(X)
            X = pca.transform(X)
            istance_transformed = pca.transform([labelledInstance.x])

        elif reductionMethod == "mca":
            import pandas as pd
            import prince

            dataK = []
            for k in range(0, len(Kneighbors_data)):
                dataK.append(Kneighbors_data[k].list)

            columnsA = [i.name for i in Kneighbors_data.domain.variables]

            if Kneighbors_data.domain.metas != ():
                for i in range(0, len(Kneighbors_data.domain.metas)):
                    columnsA.append(Kneighbors_data.domain.metas[i].name)
            data = pd.DataFrame(data=dataK, columns=columnsA)

            columnsA = [i.name for i in Kneighbors_data.domain.attributes]
            Xa = data[columnsA]
            y = Kneighbors_data.Y

            mca = prince.MCA(n_components=3, n_iter=3, copy=True,
                             check_input=True,
                             engine='auto', random_state=42)
            mca.fit(Xa)
            X = mca.transform(Xa)
            istance_transformed = mca.transform(
                [[labelledInstance[i].value for i in
                  labelledInstance.domain.attributes]])

        elif reductionMethod == "t-sne":
            from sklearn.manifold import TSNE
            XX = np.vstack([X, labelledInstance.x])
            label_istance = float(max(list(self.map_names_class.keys())) + 1)
            yy = np.concatenate((y, np.array([label_istance])))
            tsne = TSNE(n_components=2, random_state=0)
            tsne.fit(XX)
            XX = tsne.fit_transform(XX)



        else:
            print("Reduction method available: pca, t-sne, selected",
                  reductionMethod)
        label_istance = float(max(list(self.map_names_class.keys())) + 1)
        y_l = y.astype(int)
        labelMapNames = self.map_names_class.items()
        instance_label_name = self.map_names_class[int(labelledInstance.y)]

        if reductionMethod == "pca" or reductionMethod == "mca":
            XX = np.vstack([X, istance_transformed])
            yy = np.concatenate((y, np.array([label_istance])))
            ax = fig2.add_subplot(1, 2, position, projection='3d')

            # ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            sc = ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=yy, cmap="Spectral",
                            edgecolor='k')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            label_values = list(np.unique(y_l))
            label_values.append(int(label_istance))
            ax.set_title(self.classifier_name.upper())

        else:
            ax = fig2.add_subplot(1, 2, position)
            sc = ax.scatter(XX[:, 0], XX[:, 1], c=yy, cmap="tab10")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            label_values = list(np.unique(yy.astype(int)))
            ax.set_title(self.classifier_name.upper())

        colors = [sc.cmap(sc.norm(i)) for i in label_values]

        d2 = dict(labelMapNames)
        d2[int(label_istance)] = instance_label_name + "_i"
        labelMapNames_withInstance = d2.items()

        newdict = {k: dict(labelMapNames_withInstance)[k] for k in label_values}

        # ax.legend(custom_lines, [lt[1] for lt in newdict.items()],
        #          loc='center left', bbox_to_anchor=(0.9, .5), fontsize = 'x-small')

        return fig2, newdict, colors

    def showNNLocality_comparison(self, Sn_inst, fig2, position,
                                  reductionMethod="pca", training=False):
        count_inst = self.instance_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.len_dataset < small_dataset_len:
            self.starting_K = max(int(
                self.mappa_class[
                    self.map_names_class[c[0]]] * self.len_dataset),
                self.K)
        if training == True:
            Kneighbors_data, removeToDo = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, removeToDo = genNeighborsInfo(
                self.training_dataset,
                self.NearestNeighborsAll,
                self.explain_dataset[
                    count_inst], n_inst,
                self.starting_K,
                self.unique_filename,
                self.classifier,
                save=False)

        return self.visualizePoints_comparison(Sn_inst, Kneighbors_data, fig2,
                                               position, reductionMethod,
                                               training)

    def visualizePoints_comparison(self, Sn_inst, datapoints, fig2, position,
                                   reductionMethod="pca", training=False):
        from sklearn import decomposition
        count_inst = self.instance_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)

        labelledInstance = deepcopy(instTmp2)
        X = datapoints.X
        y = datapoints.Y

        if reductionMethod == "pca":
            pca = decomposition.PCA(n_components=3)
            pca.fit(X)
            X = pca.transform(X)
            istance_transformed = pca.transform([labelledInstance.x])

        elif reductionMethod == "mca":
            import pandas as pd
            import prince

            dataK = []
            for k in range(0, len(datapoints)):
                dataK.append(datapoints[k].list)

            columnsA = [i.name for i in datapoints.domain.variables]

            if datapoints.domain.metas != ():
                for i in range(0, len(datapoints.domain.metas)):
                    columnsA.append(datapoints.domain.metas[i].name)
            data = pd.DataFrame(data=dataK, columns=columnsA)

            columnsA = [i.name for i in datapoints.domain.attributes]
            Xa = data[columnsA]
            y = datapoints.Y

            mca = prince.MCA(n_components=3, n_iter=3, copy=True,
                             check_input=True,
                             engine='auto', random_state=42)
            mca.fit(Xa)
            X = mca.transform(Xa)
            istance_transformed = mca.transform(
                [[labelledInstance[i].value for i in
                  labelledInstance.domain.attributes]])

        elif reductionMethod == "t-sne":
            from sklearn.manifold import TSNE
            XX = np.vstack([X, labelledInstance.x])
            label_istance = float(max(list(self.map_names_class.keys())) + 1)
            yy = np.concatenate((y, np.array([label_istance])))
            tsne = TSNE(n_components=2, random_state=0)
            tsne.fit(XX)
            XX = tsne.fit_transform(XX)



        else:
            print("Reduction method available: pca, t-sne, selected",
                  reductionMethod)
        label_istance = float(max(list(self.map_names_class.keys())) + 1)
        y_l = y.astype(int)
        labelMapNames = self.map_names_class.items()
        instance_label_name = self.map_names_class[int(labelledInstance.y)]

        if reductionMethod == "pca" or reductionMethod == "mca":
            XX = np.vstack([X, istance_transformed])
            yy = np.concatenate((y, np.array([label_istance])))
            ax = fig2.add_subplot(1, 2, position, projection='3d')

            # ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            sc = ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=yy, cmap="Spectral",
                            edgecolor='k')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            label_values = list(np.unique(y_l))
            label_values.append(int(label_istance))
        else:
            ax = fig2.add_subplot(1, 2, position)
            sc = ax.scatter(XX[:, 0], XX[:, 1], c=yy, cmap="tab10")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            label_values = list(np.unique(yy.astype(int)))

        colors = [sc.cmap(sc.norm(i)) for i in label_values]
        custom_lines = [plt.Line2D([], [], ls="", marker='.',
                                   mec='k', mfc=c, mew=.1, ms=20) for c in
                        colors]

        d2 = dict(labelMapNames)
        d2[int(label_istance)] = instance_label_name + "_i"
        labelMapNames_withInstance = d2.items()

        newdict = {k: dict(labelMapNames_withInstance)[k] for k in label_values}

        ax.legend(custom_lines, [lt[1] for lt in newdict.items()],
                  loc='center left', bbox_to_anchor=(0.9, .5),
                  fontsize='x-small')

        return fig2

    def showExplainDatasetTabularForm(self):
        return convertOTable2Pandas(self.explain_dataset,
                                    list(map(int, self.instance_indices)))

    def showMispredictedTabularForm(self, mispred_class=False):
        sel = self.getMispredicted(mispred_class=mispred_class)
        sel_index = [self.instance_indices.index(i) for i in sel]
        return convertOTable2Pandas(self.explain_dataset, list(map(int, sel)),
                                    sel_index, self.classifier,
                                    self.map_names_class)

    def showNearestNeighTabularForm(self, Sn_inst, training=False):
        count_inst = self.instance_indices.index(Sn_inst)
        n_inst = int(Sn_inst)
        instTmp2 = Orange.data.Instance(self.explain_dataset.domain,
                                        self.explain_dataset[count_inst])
        c = self.classifier(instTmp2, False)
        small_dataset_len = 150
        if self.len_dataset < small_dataset_len:
            self.starting_K = max(int(
                self.mappa_class[
                    self.map_names_class[c[0]]] * self.len_dataset),
                self.K)
        if training == True:
            Kneighbors_data, labelledInstance = genNeighborsInfoTraining(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset.X[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier)
        else:
            Kneighbors_data, labelledInstance = genNeighborsInfo(
                self.training_dataset, self.NearestNeighborsAll,
                self.explain_dataset[count_inst], n_inst, self.starting_K,
                self.unique_filename, self.classifier, save=False)
        Kneigh_pd = convertOTable2Pandas(Kneighbors_data)
        return Kneigh_pd

    def interactiveTargetClass(self, instID):
        from ipywidgets import HBox
        classes = ["predicted"] + self.classes[:]
        w = widgets.Dropdown(
            options=classes,
            description='Target class',
            value="predicted",
            disabled=False
        )
        display(w)

        def clearAndShow(btNewObj):
            clear_output()
            display(w)
            display(h)

        def getExplainInteractiveButton(btn_object):
            if w.value == "predicted":
                rule_Index = self.explain_instance(instID)
            else:
                rule_Index = self.explain_instance(instID, w.value)

        btnTargetC = widgets.Button(description='Compute')
        btnTargetC.on_click(getExplainInteractiveButton)
        btnNewSel = widgets.Button(description='Clear')
        btnNewSel.on_click(clearAndShow)
        h = HBox([btnTargetC, btnNewSel])
        display(h)

    def interactiveMispredicted(self, mispred_class=False):
        from ipywidgets import HBox
        style = {'description_width': 'initial'}
        classes = ["All classes"] + self.classes[:]
        w = widgets.Dropdown(
            options=classes,
            description='Mispredicted classes',
            value="All classes",
            disabled=False, style=style
        )
        display(w)

        def clearAndShow(btNewObj):
            clear_output()
            display(w)
            display(h)

        def getMispredictedInteractiveButton(btn_object):
            if w.value == "All classes":
                sel = self.getMispredicted()
            else:
                sel = self.getMispredicted(mispred_class=w.value)
            sel_index = [self.instance_indices.index(i) for i in sel]
            misp = convertOTable2Pandas(self.explain_dataset,
                                        list(map(int, sel)),
                                        sel_index, self.classifier,
                                        self.map_names_class)
            from IPython.display import display
            display(misp.head())

        btnTargetC = widgets.Button(description='Get mispredicted')
        btnTargetC.on_click(getMispredictedInteractiveButton)
        btnNewSel = widgets.Button(description='Clear')
        btnNewSel.on_click(clearAndShow)
        h = HBox([btnTargetC, btnNewSel])
        display(h)


def get_KNN_threshold_max(KneighborsUser, len_dataset, thresholdError,
                          maxKNNUser):
    if KneighborsUser:
        K_NN = int(KneighborsUser)
    else:
        import math
        K_NN = int(round(math.sqrt(len_dataset)))

    if thresholdError:
        threshold = float(thresholdError)
    else:
        threshold = 0.10

    if maxKNNUser:
        maxN = int(maxKNNUser)
    else:
        maxN = getStartKValueSimplified(len_dataset)

    return K_NN, threshold, maxN
