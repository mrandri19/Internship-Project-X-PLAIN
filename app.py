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


@app.route('/instances')
def get_instances():
    if state['dataset'] is None or state['classifier'] is None:
        abort(400)

    e = XPLAIN_explainer(state['dataset'], state['classifier'], random_explain_dataset=True)
    # noinspection PyTypeChecker
    state['explainer'] = e
    assert (len(e.explain_indices) == len(e.explain_dataset))
    return jsonify([(i.list, ix) for i, ix in zip(e.explain_dataset, e.explain_indices)])


@app.route('/instance/<instance_id>', methods=['GET', 'POST'])
def get_post_instance(instance_id):
    if request.method == 'GET':
        return ""

    if request.method == 'POST':
        state['instance'] = instance_id
        return ""


@app.route('/explanation')
def get_explanation():
    e = state['explainer']
    return explanation_to_json(e.explain_instance(
        e.explain_indices[0] if state['instance'] is None else state['instance']))


def explanation_to_json(e: XPLAIN_explanation):
    return jsonify({
        'instance_id': e.instance_id,
        'diff_single': e.diff_single,
        'map_difference': e.map_difference,
        'k': e.k,
        'error': e.error,
        'target_class': e.target_class,
        'instance_class_index': e.instance_class_index,
        'prob': e.prob
    })
