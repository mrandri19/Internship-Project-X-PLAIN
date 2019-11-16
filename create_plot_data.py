

def create_plot_data(instance, single_rule_deltas, rules, instance_id, dataset_name, K, error_flag,
                     minlenname, minvalue, class_name, error, error_single,
                     classifier_name, map_difference, impo_rules_complete, pred_str,
                     save=False):
    attribute_labels = [
        f"{domain_attribute.name.capitalize()} = {str(int(attribute_value))}"
        for (domain_attribute, attribute_value)
        in zip(instance.domain.attributes, instance.x)
    ]
    attribute_impacts = single_rule_deltas

    rule_labels = [f"Rule {i}" for (i, _) in enumerate(rules, start=1)]
    rule_impacts = [map_difference[rule.replace(" ", "")] for rule in rules]

    return {"attribute_labels": attribute_labels,
            "attribute_impacts": attribute_impacts,
            "rule_labels": rule_labels,
            "rule_impacts": rule_impacts,
            "instance_id": instance_id,
            "dataset_name": dataset_name,
            "K": K,
            "error_flag": error_flag,
            "minlenname": minlenname,
            "minvalue": minvalue,
            "class_name": class_name,
            "error": error,
            "error_single": error_single,
            "classifier_name": classifier_name,
            "map_difference": map_difference,
            "impo_rules_complete": impo_rules_complete,
            "pred_str": pred_str}
