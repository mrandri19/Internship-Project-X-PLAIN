from copy import deepcopy

from flask import Flask, jsonify, abort, request
from flask_cors import CORS

from XPLAIN_class import XPLAIN_explainer
from XPLAIN_explanation_class import XPLAIN_explanation

app = Flask(__name__)
CORS(app)

classifiers = {
    'Naive Bayes': {'name': 'nb'},
    'Random Forest': {'name': 'rf'}
}

datasets = {
    'Zoo': {'file': 'zoo'},
    'Adult': {'file': 'datasets/adult_d.arff'},
    'Monks': {'file': 'datasets/monks_extended.arff'}
}

analyses = {"explain": {"display_name": "Explain the prediction"},
            "whatif": {"display_name": "What If analysis"}}

# The application's global state.
# Initialized with default values to speed up development
state = {
    'dataset': 'zoo',
    'classifier': 'nb',
    'instance': None,
    'explainer': XPLAIN_explainer('zoo', 'nb', random_explain_dataset=True),
}


# ************************************************************************************************ #

@app.route('/datasets')
def get_datasets():
    """GET /datasets returns all of the datasets"""
    return jsonify(list(datasets.keys()))


@app.route('/dataset/<name>', methods=['POST'])
def post_dataset(name):
    """POST /dataset/Zoo updates the local state setting the dataset"""
    if name not in datasets:
        abort(404)
    state['dataset'] = datasets[name]['file']
    return ""


# ************************************************************************************************ #

@app.route('/classifiers')
def get_classifers():
    """GET /classifiers returns all of the classifiers"""
    return jsonify(list(classifiers.keys()))


@app.route('/classifier/<name>', methods=['POST'])
def post_classifer(name):
    """POST /classifiers/Naive%20Bayes updates the local state setting the classifer"""
    if name not in classifiers:
        abort(404)
    state['classifier'] = classifiers[name]['name']
    return ""


# ************************************************************************************************ #

@app.route('/instances')
def get_instances():
    """
    GET /instances returns all of the instances (with their class) selected for explanation from the
    dataset which has been previously set with a POST /dataset/<name>. In this process it creates a
    new XPLAIN_explainer object, therefore reading/loading in memory the dataset.
    """
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    state['explainer'] = xp = XPLAIN_explainer(
        state['dataset'], state['classifier'], random_explain_dataset=True)

    return jsonify({
        'domain': [(attr.name, attr.values) for attr in xp.training_dataset.domain],
        'instances': [(list(instance.x) + list(instance.y), ix) for instance, ix in
                      zip(xp.explain_dataset, xp.explain_indices)]})


@app.route('/instance/<id>', methods=['POST'])
def post_instance(id):
    """POST /instance/17 updates the local state setting the instance"""
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    xp = state['explainer']
    state['instance'] = xp.explain_dataset[xp.explain_indices.index(id)]
    return ""


# ************************************************************************************************ #


@app.route('/analyses')
def get_analyses():
    """GET /analyses returns all of the analyses"""
    return jsonify(analyses)


# ************************************************************************************************ #


def explanation_to_dict(xp: XPLAIN_explanation):
    e: XPLAIN_explainer = xp.XPLAIN_explainer_o
    return {
        'instance': {attr.name: {'value': attr.values[int(value_ix)], 'options': attr.values} for
                     (attr, value_ix) in
                     zip(xp.instance.domain.attributes, xp.instance.x)},
        'domain': [(attr.name, attr.values) for attr in e.training_dataset.domain.attributes],
        'diff_single': xp.diff_single,
        'map_difference': xp.map_difference,
        'k': xp.k,
        'error': xp.error,
        'target_class': xp.target_class,
        'instance_class_index': xp.instance_class_index,
        'prob': xp.prob
    }


@app.route('/explanation')
def get_explanation():
    """
    GET /explanation returns the explanation for the instance w.r.t its class, using the explainer.
    The instance was previously set with POST /instance/<id> and belongs to the dataset set with
    POST /dataset/<name>. The explainer was created in a preceding call to POST /instance/<id>.
    """
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)
    xp = state['explainer']

    # Initialized with a default value to speed up development
    instance = xp.explain_dataset[0] if state['instance'] is None else state['instance']

    e = xp.explain_instance(instance, target_class=instance.get_class().value)
    return jsonify(explanation_to_dict(e))


# ************************************************************************************************ #

@app.route('/whatIfExplanation', methods=['GET', 'POST'])
def get_what_if_explanation():
    """
    GET /whatIfExplanation returns the what-if explanation for the instance w.r.t its class, using the
    explainer.
    The instance was previously set with POST /instance/<id> and belongs to the dataset set with
    POST /dataset/<name>. The explainer was created in a preceding call to POST /instance/<id>.

    POST /whatIfExplanation allows to perturb the instance's attributes and get an explanation for
    the perturbed instance.
    """
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)
    xp = state['explainer']

    # Initialized with a default value to speed up development
    instance = xp.explain_dataset[0] if state['instance'] is None else state['instance']

    if request.method == 'POST':
        perturbed_attributes = request.get_json(force=True)

        perturbed_instance = deepcopy(instance)
        for k, v in perturbed_attributes.items():
            perturbed_instance[k] = v['options'].index(v['value'])

        instance = perturbed_instance

    e = xp.explain_instance(instance, target_class=instance.get_class().value)
    return jsonify(
        {'explanation': explanation_to_dict(e),
         'attributes': {a.name: {'value': a.values[int(i)], 'options': a.values} for (a, i)
                        in
                        zip(instance.domain.attributes, instance.x)}})
