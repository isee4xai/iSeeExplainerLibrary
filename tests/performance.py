from resources.explainers.misc.performance import AIModelPerformance
import os
import numpy as np
from utils.base64 import bw_vector_to_base64

perform_explainer = AIModelPerformance("/Users/anjanawijekoon/projects/isee/ExplainerLibraries-aw/Models/", 
                       "")

print(perform_explainer.explain("RADIOGRAPH", {"selected_metrics":["Accuracy"]}))
print(perform_explainer.explain("RADIOGRAPH", {}))

