# noinspection PyUnresolvedReferences
import os

# noinspection PyUnresolvedReferences
from os import path
# noinspection PyUnresolvedReferences
import pickle
# noinspection PyUnresolvedReferences
from collections import Counter
# noinspection PyUnresolvedReferences
from copy import deepcopy

# noinspection PyUnresolvedReferences
import Orange

from src import DEFAULT_DIR

MAX_SAMPLE_COUNT = 100



def import_dataset(dataset_name, explain_indices, random_explain_dataset):
    if dataset_name[-4:] == "arff":
        print(dataset_name)
        dataset = loadARFF(dataset_name)
    else:
        dataset = Orange.data.Table(dataset_name)
        # TODO Eliana TMP
        if False in [i.is_discrete for i in dataset[0].domain.attributes]:
            disc = Orange.preprocess.Discretize()
            disc.method = Orange.preprocess.discretize.EqualFreq(3)
            dataset = disc(dataset)
            toARFF(dataset_name.split(".")[0] + ".arff", dataset)
            dataset = loadARFF(dataset_name.split(".")[0] + ".arff")

    dataset_len = len(dataset)
    training_indices = list(range(dataset_len))

    if random_explain_dataset:
        import random
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

    training_dataset = Orange.data.Table.from_table_rows(dataset, training_indices)
    explain_dataset = Orange.data.Table.from_table_rows(dataset, explain_indices)

    return training_dataset, explain_dataset, len(training_dataset), \
        [str(i) for i in explain_indices]


def import_datasets(dataname, n_insts, randomic):
    if dataname[-4:] == "arff":
        dataset = loadARFF(dataname)
        dataname_to_explain = dataname[:-5] + "_explain.arff"
        dataset_to_explain = loadARFF(dataname_to_explain)
    else:
        dataset = Orange.data.Table(dataname)
        dataname_to_explain = dataname[:-5] + "_explain"
        dataset_to_explain = Orange.data.Table(dataname_to_explain)
    len_dataset = len(dataset)

    len_dataset_to_explain = len(dataset_to_explain)

    if randomic:
        import random
        # 7
        random.seed(7)
        n_insts = list(random.sample(range(len_dataset_to_explain), 300))
        n_insts = [str(i) for i in n_insts]

    n_insts_int = list(map(int, n_insts))

    explain_dataset = Orange.data.Table.from_table_rows(dataset_to_explain,
                                                        n_insts_int)

    training_dataset = deepcopy(dataset)
    return training_dataset, explain_dataset, len_dataset, n_insts


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
            f.write('@attribute %s { ' % iname)
            x = []
            for j in i.values:
                s = str(j)
                if s.find(" ") == -1:
                    x.append("%s" % s)
                else:
                    x.append("'%s'" % s)
            for j in x[:-1]:
                f.write('%s,' % j)
            f.write('%s }\n' % x[-1])
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


def loadARFF_Weka(filename):
    if not os.path.exists(filename) and os.path.exists(filename + ".arff"):
        filename = filename + ".arff"
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

        return table


def loadARFF(filename):
    """Return class:`Orange.data.Table` containing data from file in Weka ARFF format
       if there exists no .xml file with the same name. If it does, a multi-label
       dataset is read and returned.
    """
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    if os.path.exists(filename + ".xml") and os.path.exists(filename + ".arff"):
        xml_name = filename + ".xml"
        arff_name = filename + ".arff"
        return Orange.multilabel.mulan.trans_mulan_data(xml_name, arff_name)
    else:
        return loadARFF_Weka(filename)


def printTree(classifier, name):
    features_names = get_features_names(classifier)
    from io import StringIO
    import pydotplus
    dot_data = StringIO()
    from sklearn import tree
    if features_names != None:
        tree.export_graphviz(classifier.skl_model, out_file=dot_data,
                             feature_names=features_names, filled=True,
                             rounded=True, special_characters=True)
    else:
        tree.export_graphviz(classifier.skl_model, out_file=dot_data,
                             filled=True, rounded=True,
                             special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(name + "_tree.pdf")


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


def useExistingModel_v2(classif, classifierparameter, dataname):
    import os
    if os.path.exists(DEFAULT_DIR + "models") == False:
        os.makedirs(DEFAULT_DIR + "models")
    m = ""
    if classifierparameter != None:
        m = "-" + classifierparameter
    file_path = DEFAULT_DIR + "models/" + dataname + "-" + classifierparameter + m
    if (os.path.exists(file_path) == True):
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        if classif == "tree":
            modelname = "<class 'Orange.classification.tree.SklTreeClassifier'>"
        elif classif == "nb":
            modelname = "<class 'Orange.classification.naive_bayes.NaiveBayesModel'>"
        elif classif == "nn":
            modelname = "<class 'Orange.classification.base_classification.SklModelClassification'>"
        elif classif == "knn":
            modelname = "<class 'Orange.classification.base_classification.SklModelClassification'>"
        elif classif == "rf":
            modelname = "<class 'Orange.classification.random_forest.RandomForestClassifier'>"
        else:
            return False

        if str(type(model)) == modelname:
            return model

    return False


def getClassifier_v2(training_dataset, classif, classifierparameter, exit):
    classif = classif
    classifier = None
    reason = ""
    if classif == "tree":
        if (classifierparameter == None):
            measure = "gini"
        else:
            measure = classifierparameter.split("-")[0]
        if (measure) != "gini" and (measure) != "entropy":
            measure = "entropy"
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnertree = Orange.classification.SklTreeLearner(
            preprocessors=continuizer, max_depth=7, min_samples_split=5,
            min_samples_leaf=3, random_state=1)
        # learnertree=Orange.classification.SklTreeLearner(preprocessors=continuizer, random_state=1)

        classifier = learnertree(training_dataset)

        printTree(classifier, training_dataset.name)
    elif classif == "nb":
        learnernb = Orange.classification.NaiveBayesLearner()
        classifier = learnernb(training_dataset)
    elif classif == "nn":
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnernet = Orange.classification.NNClassificationLearner(
            preprocessors=continuizer, random_state=42,
            max_iter=1000)

        classifier = learnernet(training_dataset)
    elif classif == "rf":
        import random
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnerrf = Orange.classification.RandomForestLearner(
            preprocessors=continuizer, random_state=42)
        classifier = learnerrf(training_dataset)
    elif classif == "svm":
        import random
        continuizer = Orange.preprocess.Continuize()
        continuizer.multinomial_treatment = continuizer.Indicators
        learnerrf = Orange.classification.SVMLearner(preprocessors=continuizer)
        classifier = learnerrf(training_dataset)
    elif classif == "knn":
        if classifierparameter == None:
            exit = 1
            reason = "k - missing the K parameter"
        elif (len(classifierparameter.split("-")) == 1):
            KofKNN = int(classifierparameter.split("-")[0])
            distance = ""
        else:
            KofKNN = int(classifierparameter.split("-")[0])
            distance = classifierparameter.split("-")[1]
        if exit != 1:
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
        reason = "Classification model not available"
        exit = 1

    return classifier, exit, reason


def createDir(outdir):
    try:
        os.makedirs(outdir)
    except:
        pass


def gen_neighbors_info(training_dataset, NearestNeighborsAll, instance, k,
                       unique_filename, classifier, save=True):
    instance_features = instance.x
    nearest_neighbors = NearestNeighborsAll.kneighbors([instance_features], k,
                                                       return_distance=False)

    out_data_raw = []
    lendataset_nearest_neighbors = len(nearest_neighbors[0])
    for i in range(0, lendataset_nearest_neighbors):
        c = classifier(training_dataset[nearest_neighbors[0][i]])
        instanceK = Orange.data.Instance(training_dataset.domain,
                                         training_dataset[
                                             nearest_neighbors[0][i]])
        instanceK.set_class(c[0])
        if i == 0:
            instanceK_i = Orange.data.Instance(training_dataset.domain,
                                               instance)
            c = classifier(instanceK_i)
            instanceTmp = deepcopy(instanceK_i)
            instanceTmp.set_class(c[0])
            out_data_raw.append(instanceTmp)
        out_data_raw.append(instanceK)

    out_data = Orange.data.Table(training_dataset.domain, out_data_raw)

    c = classifier(training_dataset[nearest_neighbors[0][0]])
    instance0 = Orange.data.Instance(training_dataset.domain,
                                     training_dataset[nearest_neighbors[0][0]])
    instance0.set_class(c[0])
    out_data1 = Orange.data.Table(training_dataset.domain, [instance0])

    if save:
        import os
        path = DEFAULT_DIR + unique_filename
        if not os.path.exists(path):
            os.makedirs(path)
        toARFF(path + "/Knnres.arff", out_data)
        toARFF(path + "/Filetest.arff", out_data1)
        toARFF(path + "/gen-k0.arff", out_data1)

    return out_data, out_data1


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


def compute_prediction_difference_subset(training_dataset,
                                         instance,
                                         rule_body_indices,
                                         classifier,
                                         instance_class_index,
                                         instance_predictions_cache):
    """
    Compute the prediction difference for an instance in a training_dataset, w.r.t. some
    rules and a class, given a classifier
    """
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
def compute_prediction_difference_single(instT, classifier, indexI, dataset):
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


def computeMappaClass_b(data):
    mappa_class2 = {}
    h = len(data)
    dim_d = len(data[0])
    for d in data[:]:
        c_tmp = d[dim_d - 1].value
        if c_tmp in mappa_class2:
            mappa_class2[c_tmp] = mappa_class2[c_tmp] + 1.0
        else:
            mappa_class2[c_tmp] = 1.0

    for key in mappa_class2.keys():
        mappa_class2[key] = mappa_class2[key] / h

    return mappa_class2


def convertOTable2Pandas(orangeTable, ids=None, sel="all", cl=None, mapName=None):
    import pandas as pd

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
