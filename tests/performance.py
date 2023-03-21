from resources.explainers.misc.performance import AIModelPerformance
# import os
# import numpy as np

perform_explainer = AIModelPerformance("folder-path", 
                       "")

print(perform_explainer.explain("model-id", {"selected_metrics":["Accuracy"]}))
print(perform_explainer.explain("model-id", {}))

