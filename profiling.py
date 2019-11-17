#%%
from XPLAIN_class import XPLAIN_explainer

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

dataset = "datasets/adult_d.arff"
# dataset = "zoo"
classifier = "rf"
instance_id = "0"
#%%
import time

total_running_time_start = time.time()

explainer = XPLAIN_explainer(dataset, [], classifier, trainExplainSet=True)

print("Total running time: %s" % (time.time() - total_running_time_start))
#%%

explanation_start = time.time()
explainer.getExplanation_i(instance_id)
plt.close()
plt.close()
print("explanation time: %s" % (time.time() - explanation_start))
