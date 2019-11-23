import functools
import operator
from collections import Counter
from copy import deepcopy

import Orange


def computePredictionDifferenceSubsetRandomOnlyExisting(locality_dataset,
                                                        instance,
                                                        rule_bodies_indices,
                                                        classifier,
                                                        instance_class_index):
    print("computePredictionDifferenceSubsetRandomOnlyExisting")
    print("instT=", instance)
    print("inputA=", rule_bodies_indices)

    difference_map = {}

    inst = deepcopy(instance)
    attribute_body_indices_map = {}

    rule_bodies_attributes = []
    for rule_body_indices in rule_bodies_indices:
        rule_body_attributes = [
            locality_dataset.domain.attributes[rule_body_index - 1] for
            rule_body_index in rule_body_indices]
        if (len(rule_body_attributes)) > 1:
            rule_bodies_attributes.append(rule_body_attributes)
            attribute_body_indices_map[
                ','.join(map(str, rule_body_attributes))] = ','.join(
                map(str, rule_body_indices))

    for g_v2 in rule_bodies_attributes:
        d = Orange.data.Table()
        g_v2_a_i = Orange.data.Domain(g_v2)
        filtered_i = d.from_table(g_v2_a_i, locality_dataset)
        c = Counter(map(tuple, filtered_i.X))
        freq = dict(c.items())

        for k_ex in freq:
            inst1 = deepcopy(instance)
            for v in range(0, len(g_v2)):
                inst1[g_v2_a_i[v]] = k_ex[v]

            name = [j.name for j in g_v2]
            combinations = functools.reduce(operator.mul,
                                            [len(i.values) for i in g_v2])

            setIndex = attribute_body_indices_map[','.join(map(str, name))]
            if setIndex not in difference_map:
                difference_map[setIndex] = 0.0
            c0 = classifier(inst1, False)[0]
            c1 = classifier(inst1, True)[0]

            prob = 0.000000
            prob = c1[instance_class_index]
            newvalue = prob * freq[k_ex] / len(locality_dataset)
            difference_map[setIndex] = difference_map[setIndex] + newvalue

    c0 = classifier(inst, False)[0]
    c1 = classifier(inst, True)[0]

    prob = c1[instance_class_index]

    for key, value in difference_map.items():
        difference_map[key] = prob - difference_map[key]

    return difference_map


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
