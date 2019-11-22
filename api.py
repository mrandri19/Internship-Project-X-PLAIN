from XPLAIN_class import XPLAIN_explainer


def get_explanation(dataset, classifier):
    explainer = XPLAIN_explainer(dataset, classifier, train_explain_set=True)
    return explainer.explain_instance(explainer.instance_indices[0])


if __name__ == "__main__":
    get_explanation("datasets/adult_d.arff","nb")
