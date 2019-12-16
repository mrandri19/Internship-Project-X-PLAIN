from copy import deepcopy

from flask import Flask, jsonify, abort, request
from flask_cors import CORS

from XPLAIN_class import import_datasets, import_dataset, XPLAIN_explainer
from XPLAIN_explanation_class import XPLAIN_explanation

app = Flask(__name__)
CORS(app)

# TODO: use sessions and local cache (redis?)
# TODO: write tests instead of this
state = {
    'dataset': 'zoo',
    'classifier': 'nb',
    'instance': None,
    'explainer': XPLAIN_explainer('zoo', 'nb', random_explain_dataset=True),
    'cached_explanations': {}
}

datasets = {
    'Zoo': {'file': 'zoo'},
    'Adult': {'file': 'datasets/adult_d.arff'},
    'Monks': {'file': 'datasets/monks_extended.arff'}
}


@app.route('/datasets')
def get_datasets():
    return jsonify(list(datasets.keys()))


@app.route('/dataset/<name>', methods=['GET', 'POST'])
def get_post_dataset(name):
    if name not in datasets:
        abort(404)
    file = datasets[name]['file']

    if request.method == 'GET':
        if file == "datasets/adult_d.arff" \
                or file == "datasets/compas-scores-two-years_d.arff":
            train_dataset, _, _, _ = import_datasets(file, [], False, False)
        else:
            train_dataset, _, _, _ = import_dataset(file, [], False)

        return jsonify({v.name: v.values for v in train_dataset.domain.variables})

    if request.method == 'POST':
        state['dataset'] = file
        return ""


classifiers = {
    'Naive Bayes': {'name': 'nb'},
    'Random Forest': {'name': 'rf'}
}


@app.route('/classifiers')
def get_classifers():
    return jsonify(list(classifiers.keys()))


@app.route('/classifier/<name>', methods=['GET', 'POST'])
def get_post_classifer(name):
    if name not in classifiers:
        abort(404)

    if request.method == 'GET':
        return classifiers[name]

    if request.method == 'POST':
        state['classifier'] = classifiers[name]['name']
        return ""


# TODO(andrea): cache
@app.route('/instances')
def get_instances():
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    e = XPLAIN_explainer(
        state['dataset'], state['classifier'], random_explain_dataset=True)
    # noinspection PyTypeChecker
    state['explainer'] = e
    assert (len(e.explain_indices) == len(e.explain_dataset))
    return jsonify({
        'domain': [(a.name, a.values) for a in e.training_dataset.domain],
        'instances': [(list(i.x) + list(i.y), ix) for i, ix in zip(e.explain_dataset, e.explain_indices)]})


@app.route('/instance/<instance_id>', methods=['POST'])
def get_post_instance(instance_id):
    if request.method == 'POST':
        print(instance_id)
        # TODO(andrea): fix
        state['instance'] = state['explainer'].explain_dataset[0]
        return ""


analyses = [{"id": "explain", "display_name": "Explain the prediction"}, {
    "id": "whatif", "display_name": "What If analysis"}]


@app.route('/analyses')
def get_analyses():
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)
    return jsonify(analyses)


@app.route('/explanation')
def get_explanation():
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)
    e = state['explainer']
    instance = e.explain_dataset[0] if state['instance'] is None else state['instance']
    return jsonify(explanation_to_dict(e.explain_instance(instance, target_class=instance.get_class().value)))


@app.route('/whatIfExplanation', methods=['GET', 'POST'])
def get_what_if_explanation():
    if state['dataset'] is None or state['classifier'] is None or state['explainer'] is None:
        abort(400)

    e = state['explainer']
    instance = e.explain_dataset[0] if state['instance'] is None else state['instance']

    if request.method == 'GET':
        return jsonify(
            {'explanation': explanation_to_dict(e.explain_instance(instance, target_class=instance.get_class().value)),
             'attributes': {a.name: {'value': a.values[int(i)], 'options': a.values} for (a, i) in
                            zip(instance.domain.attributes, instance.x)}})
    if request.method == 'POST':
        perturbed_attributes = request.get_json(force=True)
        print(perturbed_attributes)

        perturbed_instance = deepcopy(instance)
        for k, v in perturbed_attributes.items():
            perturbed_instance[k] = v['options'].index(v['value'])

        return jsonify(
            {'explanation': explanation_to_dict(
                e.explain_instance(perturbed_instance, target_class=perturbed_instance.get_class().value)),
                'attributes': {a.name: {'value': a.values[int(i)], 'options': a.values} for (a, i) in
                               zip(perturbed_instance.domain.attributes, perturbed_instance.x)}})


# TODO: this could be an idea for the next api iteration
# @app.route('/datasets/<dataset_id>/instances')
# def get_explanation(dataset_id):
#     print(dataset_id)
#     return ""

def explanation_to_dict(xp: XPLAIN_explanation):
    e: XPLAIN_explainer = xp.XPLAIN_explainer_o
    return {
        'instance': {a.name: {'value': a.values[int(i)], 'options': a.values} for (a, i) in
                     zip(xp.instance.domain.attributes, xp.instance.x)},
        'domain': [(a.name, a.values) for a in e.training_dataset.domain.attributes],
        'diff_single': xp.diff_single,
        'map_difference': xp.map_difference,
        'k': xp.k,
        'error': xp.error,
        'target_class': xp.target_class,
        'instance_class_index': xp.instance_class_index,
        'prob': xp.prob
    }
