from collections import Counter
from copy import deepcopy

import Orange


def compute_prediction_difference_subset_random_only_existing(training_dataset,
                                                              instance,
                                                              rule_bodies_indices,
                                                              classifier,
                                                              instance_class_index):
    print("computePredictionDifferenceSubsetRandomOnlyExisting")
    print("instance=", instance)
    print("rule_bodies_indices=", rule_bodies_indices)

    prediction_differences = {}

    rule_bodies_attributes, attribute_body_indices_map = \
        replace_rule_bodies_indices_with_attributes(
            training_dataset, rule_bodies_indices)

    # For each rule
    for rule_body_attributes in rule_bodies_attributes:
        # Take only the considered attributes from the dataset
        rule_domain = Orange.data.Domain(rule_body_attributes)
        filtered_dataset = Orange.data.Table().from_table(rule_domain, training_dataset)

        # Count how many times a set of attribute values appears in the dataset
        attribute_sets_appearances = dict(
            Counter(map(tuple, filtered_dataset.X)).items())

        print("len(rule_body_attributes)=", len(rule_body_attributes),
              " <=> len(attribute_sets_appearances)=", len(attribute_sets_appearances))

        # For each set of attributes
        for (attribute_set_key,
             appearances) in attribute_sets_appearances.items():

            # Take the original instance and replace a subset of its attributes
            # with the attribute set
            perturbed_instance = deepcopy(instance)
            for i in range(len(rule_body_attributes)):
                perturbed_instance[rule_domain[i]] = attribute_set_key[i]

            # key of the set of attributes in the prediction difference map
            attribute_set_key = attribute_body_indices_map[
                ','.join([attribute.name for attribute
                          in rule_body_attributes])]

            # if this set of attributes has not been considered yet, initialize
            # its prediction difference to 0
            if attribute_set_key not in prediction_differences:
                prediction_differences[attribute_set_key] = 0.0

            # Probability that the perturbed instance belongs to a certain class
            prob = classifier(perturbed_instance, True)[0][instance_class_index]

            # Update the prediction difference using the weighted average of the
            # probability over the frequency of this attribute set in the
            # dataset
            prediction_differences[attribute_set_key] += (
                    prob * appearances / len(training_dataset)
            )

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = classifier(instance, True)[0][instance_class_index]

    for attribute_set_key in prediction_differences:
        prediction_differences[attribute_set_key] = p - prediction_differences[
            attribute_set_key]

    return prediction_differences


def replace_rule_bodies_indices_with_attributes(locality_dataset,
                                                rule_bodies_indices,
                                                ):
    """
    Converts
    [[1, 2, 3, 4, 8, 9, 10, 11]]
    into
    [[DiscreteVariable(name='hair', values=['0', '1']),
      DiscreteVariable(name='feathers', values=['0', '1']),
      DiscreteVariable(name='eggs', values=['0', '1']),
      DiscreteVariable(name='milk', values=['0', '1']),
      DiscreteVariable(name='toothed', values=['0', '1']),
      DiscreteVariable(name='backbone', values=['0', '1']),
      DiscreteVariable(name='breathes', values=['0', '1']),
      DiscreteVariable(name='venomous', values=['0', '1'])]]
    :param locality_dataset:
    :param rule_bodies_indices:
    :return:
    """
    attribute_body_indices_map = {}

    rule_bodies_attributes = []
    for rule_body_indices in rule_bodies_indices:
        rule_body_attributes = [
            locality_dataset.domain.attributes[rule_body_index - 1] for
            rule_body_index in rule_body_indices]
        if len(rule_body_attributes) > 1:
            rule_bodies_attributes.append(rule_body_attributes)
            attribute_body_indices_map[
                ','.join(map(str, rule_body_attributes))] = ','.join(
                map(str, rule_body_indices))
    return rule_bodies_attributes, attribute_body_indices_map


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
