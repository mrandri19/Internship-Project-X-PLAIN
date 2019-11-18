#!/usr/bin/env python -W ignore::DeprecationWarning
from copy import deepcopy
from collections import Counter
import operator
import functools
import Orange
from XPLAIN_class import *
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def computePredictionDifferenceSubsetRandomOnlyExisting(dataset, instT, inputA,
                                                        classname, classifier,
                                                        indexI, mappa_sum_v2):
    computePredictionDifferenceSubsetRandomOnlyExisting_start = time.time()
    print("computePredictionDifferenceSubsetRandomOnlyExisting")
    print(inputA)

    inst = deepcopy(instT)
    mappa_attr_index = {}

    listaval_v2 = []
    for v1 in inputA[:]:
        listaval1 = []
        for v2 in v1[:]:
            n = 0
            for k in dataset.domain.attributes[:]:
                n = n + 1
                if n == v2:
                    listaval1.append(k)
        if (len(listaval1)) > 1:
            listaval_v2.append(listaval1)
            mappa_attr_index[','.join(map(str, listaval1))] = ','.join(
                map(str, v1))

    for g_v2 in listaval_v2[:]:
        d = Orange.data.Table()
        g_v2_a_i = Orange.data.Domain(g_v2)
        filtered_i = d.from_table(g_v2_a_i, dataset)
        c = Counter(map(tuple, filtered_i.X))
        freq = dict(c.items())

        for k_ex in freq:
            inst1 = deepcopy(instT)
            for v in range(0, len(g_v2)):
                inst1[g_v2_a_i[v]] = k_ex[v]

            name = [j.name for j in g_v2]
            combinations = functools.reduce(operator.mul,
                                            [len(i.values) for i in g_v2])

            setIndex = mappa_attr_index[','.join(map(str, name))]
            if setIndex not in mappa_sum_v2:
                mappa_sum_v2[setIndex] = 0.0
            c0 = classifier(inst1, False)[0]
            c1 = classifier(inst1, True)[0]

            prob = 0.000000
            prob = c1[indexI]
            newvalue = prob * freq[k_ex] / len(dataset)
            mappa_sum_v2[setIndex] = mappa_sum_v2[setIndex] + newvalue

    c0 = classifier(inst, False)[0]
    c1 = classifier(inst, True)[0]

    prob = c1[indexI]

    for key, value in mappa_sum_v2.items():
        mappa_sum_v2[key] = prob - mappa_sum_v2[key]

    print("computePredictionDifferenceSinglever2 time: %s" % (
        time.time() - computePredictionDifferenceSubsetRandomOnlyExisting_start))
    return mappa_sum_v2


# Single explanation. Change 1 value at the time e compute the difference
def computePredictionDifferenceSinglever2(instT, classifier, indexI, dataset):
    from copy import deepcopy
    i = deepcopy(instT)
    listaoutput = []

    c0 = classifier(i, False)[0]
    c1 = classifier(i, True)[0]
    prob = c1[indexI]

    for u in i.domain.attributes[:]:
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

            prob = 0.000000
            prob = c1[indexI]
            test = freq[k_ex] / len(dataset)
            # newvalue=prob*freq[k_ex]/len(dataset)
            newvalue = prob * test
            listaoutput[t] = listaoutput[t] + newvalue

    l = len(listaoutput)

    for i in range(0, l):
        listaoutput[i] = prob - listaoutput[i]
    return listaoutput
