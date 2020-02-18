# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import pickle
import random
# noinspection PyUnresolvedReferences
from collections import Counter
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
OT = 0
MT = 1

from collections import defaultdict


class Dataset:
    _df: pd.DataFrame

    def __init__(self, data, attributes):
        self._df = pd.DataFrame(data)
        self.attributes = attributes

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
        return self.attributes[-1][1]

    def X(self):
        """The dataset's attributes as a numpy float64 array."""
        return self._encoded_df.iloc[:, :-1].to_numpy().astype(np.float64)

    def row_inverse_transform_value(self, attr, column_index):
        return self._column_encoders[column_index].inverse_transform(attr)

    def class_column_name(self):
        """"The column name of the class attribute"""
        return self.attributes[-1][0]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, item):
        return self._encoded_df.iloc[item]


def assert_orange_pd_equal(table: Orange.data.Table, dataset: Dataset):
    # TODO(Andrea): Remove when Orange is completely out
    assert len(table) == len(dataset._df)


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

        dataset_file_name = join(DEFAULT_DIR, "datasets", dataset_name.split(".")[0]) + ".arff"
        toARFF(dataset_file_name, orange_dataset)

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
    pd_training_dataset = Dataset(pd_dataset._df.iloc[training_indices], pd_dataset.attributes)
    assert_orange_pd_equal(orange_training_dataset, pd_training_dataset)

    orange_explain_dataset = Orange.data.Table.from_table_rows(orange_dataset, explain_indices)
    pd_explain_dataset = Dataset(pd_dataset._df.iloc[explain_indices], pd_dataset.attributes)
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
                                 pd_explain_dataset.attributes)
    assert_orange_pd_equal(orange_explain_dataset, pd_explain_dataset)

    return (orange_training_dataset, pd_training_dataset), (
        orange_explain_dataset, pd_explain_dataset), len_dataset, [str(i) for i in explain_indices]


def toARFF(filename, table, try_numericize=0):
    """Save class:`Orange.data.Table` to file in Weka's ARFF format"""
    t = table
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    # print( filename
    f = open(filename + '.arff', 'w')
    f.write('@relation %s\n' % t.domain.class_var.name)
    # attributes
    ats = [i for i in t.domain.attributes]
    ats.append(t.domain.class_var)
    for i in ats:
        real = 1
        if i.is_discrete == 1:
            if try_numericize:
                # try if all values numeric
                for j in i.values:
                    try:
                        x = float(j)
                    except:
                        real = 0  # failed
                        break
            else:
                real = 0
        iname = str(i.name)
        if iname.find(" ") != -1:
            iname = "'%s'" % iname
        if real == 1:
            f.write('@attribute %s real\n' % iname)
        else:
            f.write('@attribute %s {' % iname)
            x = []
            for j in i.values:
                s = str(j)
                if s.find(" ") == -1:
                    x.append("%s" % s)
                else:
                    x.append("'%s'" % s)
            for j in x[:-1]:
                f.write('%s,' % j)
            f.write('%s}\n' % x[-1])
    f.write('@data\n')
    for j in t:
        x = []
        for i in range(len(ats)):
            s = str(j[i])
            if s.find(" ") == -1:
                x.append("%s" % s)
            else:
                x.append("'%s'" % s)
        for i in x[:-1]:
            f.write('%s,' % i)
        f.write('%s\n' % x[-1])
    f.close()


def loadARFF_Weka(filename: str) -> Tuple[Orange.data.Table, Dataset]:
    with open(filename, 'r') as f:

        attributes = []
        name = ''
        in_header = False  # header
        rows = []

        for line in f.readlines():
            line = line.rstrip("\n\r")  # strip trailing whitespace
            line = line.replace('\t', ' ')  # get rid of tabs
            line = line.split('%')[0]  # strip comments
            if len(line.strip()) == 0:  # ignore empty lines
                continue
            if not in_header and line[0] != '@':
                print(("ARFF import ignoring:", line))
            if in_header:  # Header
                if line[0] == '{':  # sparse data format, begin with '{', ends with '}'
                    r = [None] * len(attributes)
                    row = line[1:-1]
                    row = row.split(',')
                    for xs in row:
                        y = xs.split(" ")
                        if len(y) != 2:
                            raise ValueError("the format of the data is error")
                        # noinspection PyTypeChecker
                        r[int(y[0])] = y[1]
                    rows.append(r)
                else:  # normal data format, split by ','
                    row = line.split(',')
                    r = []
                    for xs in row:
                        y = xs.strip(" ")
                        if len(y) > 0:
                            if y[0] == "'" or y[0] == '"':
                                r.append(xs.strip("'\""))
                            else:
                                ns = xs.split()
                                for ls in ns:
                                    if len(ls) > 0:
                                        r.append(ls)
                        else:
                            r.append('?')
                    rows.append(r[:len(attributes)])
            else:  # Data
                y = []
                for cy in line.split(' '):
                    if len(cy) > 0:
                        y.append(cy)
                if str.lower(y[0][1:]) == 'data':
                    in_header = True
                elif str.lower(y[0][1:]) == 'relation':
                    name = str.strip(y[1])
                elif str.lower(y[0][1:]) == 'attribute':
                    if y[1][0] == "'":
                        atn = y[1].strip("' ")
                        idx = 1
                        while y[idx][-1] != "'":
                            idx += 1
                            atn += ' ' + y[idx]
                        atn = atn.strip("' ")
                    else:
                        atn = y[1]
                    z = line.split('{')
                    w = z[-1].split('}')
                    if len(z) > 1 and len(w) > 1:
                        # there is a list of values
                        vals = []
                        for y in w[0].split(','):
                            sy = y.strip(" '\"")
                            if len(sy) > 0:
                                vals.append(sy)
                        a = Orange.data.DiscreteVariable.make(atn, vals, True, 0)
                    else:
                        a = Orange.data.variable.ContinuousVariable.make(atn)
                    attributes.append(a)

        # generate the domain
        if attributes[-1].name == name:
            domain = Orange.data.Domain(attributes[:-1], attributes[-1])
        else:
            new_attr = []
            for att in attributes:
                if att != name:
                    new_attr.append(att)
            domain = Orange.data.Domain(new_attr)

        instances = [Orange.data.Instance(domain, row) for row in rows]

        table = Orange.data.Table.from_list(domain, instances)
        table.name = name

        f.seek(0)
        a = arff.load(f)
        dataset = Dataset(a['data'], a['attributes'])

        assert_orange_pd_equal(table, dataset)

        return table, dataset


def loadARFF(filename: str) -> Tuple[Orange.data.Table, Dataset]:
    return loadARFF_Weka(filename)


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
                   classifier_parameter: str, should_exit) -> Tuple[
    Orange.classification.Learner, bool, str]:
    # TODO(Andrea): FINISH continue threading through the pandas dataset
    classifier_name = classifier_name
    classifier = None
    exit_reason = ""

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
            should_exit = True
            exit_reason = "k - missing the K parameter"
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
        exit_reason = "Classification model not available"
        should_exit = True

    return classifier, should_exit, exit_reason


def createDir(outdir):
    try:
        os.makedirs(outdir)
    except:
        pass


def gen_neighbors_info(training_dataset_, nbrs, instance, k,
                       unique_filename, classifier):
    training_dataset = training_dataset_[OT]
    domain = training_dataset.domain
    nearest_neighbors_ixs = nbrs.kneighbors([instance.x], k,
                                            return_distance=False)[0]
    closest_instance = training_dataset[nearest_neighbors_ixs[0]]

    classified_instance = deepcopy(instance)
    classified_instance.set_class(classifier(instance)[0])
    classified_instances = [classified_instance]

    for neigh_ix in nearest_neighbors_ixs:
        neigh = training_dataset[neigh_ix]
        classified_neigh = deepcopy(neigh)
        classified_neigh.set_class(classifier(neigh)[0])

        classified_instances.append(classified_neigh)

    classified_instances_table = Orange.data.Table(domain, classified_instances)

    closest_instance_classified = deepcopy(closest_instance)
    closest_instance_classified.set_class(classifier(closest_instance)[0])
    closest_instance_table = Orange.data.Table(domain, [closest_instance_classified])

    import os
    p = DEFAULT_DIR + unique_filename
    if not os.path.exists(p):
        os.makedirs(p)
    toARFF(p + "/Knnres.arff", classified_instances_table)
    toARFF(p + "/Filetest.arff", closest_instance_table)


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
    training_dataset = training_dataset_[OT]

    rule_attributes = [
        training_dataset.domain.attributes[rule_body_index - 1] for
        rule_body_index in rule_body_indices]

    # Take only the considered attributes from the dataset
    rule_domain = Orange.data.Domain(rule_attributes)
    filtered_dataset = Orange.data.Table().from_table(rule_domain, training_dataset)

    # Count how many times a set of attribute values appears in the dataset
    attribute_sets_occurrences = dict(
        Counter(map(tuple, filtered_dataset.X)).items())

    # For each set of attributes
    differences = [compute_perturbed_difference(item, classifier, instance, instance_class_index,
                                                rule_attributes, rule_domain, training_dataset) for
                   item in
                   attribute_sets_occurrences.items()]

    prediction_difference = sum(differences)

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = classifier(instance, True)[0][instance_class_index]
    prediction_differences = p - prediction_difference

    return prediction_differences


def compute_perturbed_difference(item, classifier, instance, instance_class_index,
                                 rule_attributes, rule_domain, training_dataset):
    (attribute_set, occurrences) = item
    perturbed_instance = Orange.data.Instance(training_dataset.domain, instance.list)
    for i in range(len(rule_attributes)):
        perturbed_instance[rule_domain[i]] = attribute_set[i]
    # cache_key = tuple(perturbed_instance.x)
    # if cache_key not in instance_predictions_cache:
    #     instance_predictions_cache[cache_key] = classifier(perturbed_instance, True)[0][
    #         instance_class_index]
    # prob = instance_predictions_cache[cache_key]
    prob = classifier(perturbed_instance, True)[0][instance_class_index]

    # Compute the prediction difference using the weighted average of the
    # probability over the frequency of this attribute set in the
    # dataset
    difference = prob * occurrences / len(training_dataset)
    return difference


# Single explanation. Change 1 value at the time e compute the difference
def compute_prediction_difference_single(instT, classifier, indexI, dataset_):
    dataset = dataset_[OT]
    from copy import deepcopy
    i = deepcopy(instT)
    listaoutput = []

    c1 = classifier(i, True)[0]
    prob = c1[indexI]

    for _ in i.domain.attributes[:]:
        listaoutput.append(0.0)

    t = -1
    for k in dataset.domain.attributes[:]:
        d = Orange.data.Table()
        t = t + 1
        k_a_i = Orange.data.Domain([k])
        filtered_i = d.from_table(k_a_i, dataset)
        c = Counter(map(tuple, filtered_i.X))
        freq = dict(c.items())

        for k_ex in freq:
            inst1 = deepcopy(instT)
            inst1[k] = k_ex[0]
            c1 = classifier(inst1, True)[0]

            prob = c1[indexI]
            test = freq[k_ex] / len(dataset)
            # newvalue=prob*freq[k_ex]/len(dataset)
            newvalue = prob * test
            listaoutput[t] = listaoutput[t] + newvalue

    l = len(listaoutput)

    for i in range(0, l):
        listaoutput[i] = prob - listaoutput[i]
    return listaoutput


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


def convertOTable2Pandas(orangeTable, ids=None, sel="all", cl=None, mapName=None):
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
    createDir(dirO)
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
