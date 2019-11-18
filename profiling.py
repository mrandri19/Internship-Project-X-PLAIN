import time
import matplotlib.pyplot as plt
from XPLAIN_class import XPLAIN_explainer

import matplotlib as mpl
mpl.use('Agg')

dataset = "datasets/adult_d.arff"
# dataset = "zoo"
classifier = "rf"
instance_id = "0"

explainer_creation_time_start = time.time()
explainer = XPLAIN_explainer(dataset, [], classifier, trainExplainSet=True)
print("explainer creation time: %s" %
      (time.time() - explainer_creation_time_start))

explanation_start = time.time()
explainer.getExplanation_i(instance_id)
plt.close()
plt.close()
print("explanation time: %s" % (time.time() - explanation_start))

# [[1, 7, 9], [1, 9, 10], [4, 9, 10], [1, 4, 7, 9, 10]]

# [[Female, Private, Bachelors, White, Never-married, ... | <=50K],
#  [Male, Private, Associates, White, Never-married, ... | <=50K],
#  [Male, Local-gov, High-School-grad, White, Married, ... | <=50K],
#  [Male, Private, Doctorate, White, Married, ... | >50K],
#  [Male, Private, High-School-grad, White, Married, ... | <=50K],
#  ...
# ]

# RandomForestClassifier(skl_model=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=None, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#                        oob_score=False, random_state=42, verbose=0,
#                        warm_start=False))  # params={'n_estimators': 10, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': 'auto', 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': False, 'n_jobs': 1, 'random_state': 42, 'verbose': 0, 'class_weight': None}

# [Female, Private, Dropout, Black, Widowed, ... | <=50K]
