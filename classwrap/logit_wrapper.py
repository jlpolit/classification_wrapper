from scipy.stats import uniform

from .base_wrapper import ClassificationWrapper


class LogitWrapper(ClassificationWrapper):

    def _build_search_space(self):
        self.param_search_space = {
            'preprocessor__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
            'classifier__C': [0.1, 1.0, 10, 100],
            'classifier__l1_ratio': uniform(0, 1)
        }

