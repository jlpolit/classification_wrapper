from scipy.stats import uniform

from .base_wrapper import ClassificationWrapper


# TODO: test!
class GradientBoostWrapper(ClassificationWrapper):
    def __init__(self, model_type="gbm", random_state=None, early_stopping=False):
        self.early_stopping = early_stopping
        super().__init__(model_type, random_state)

    def _build_search_space(self):
        self.param_search_space = {
            'preprocessor__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
            'classifier__learning_rate': [0.1, 0.01, 0.001],
            'classifier__n_estimators': uniform(50, 1000),
            'classifier__subsample': uniform(.4, 1),
            'classifier__max_depth': uniform(2, 10)
        }
        if self.early_stopping:
            self.param_search_space['validation_fraction'] = uniform(0.1, 0.5, 0.1)
            self.param_search_space['n_iter_no_change'] = uniform(10, 100)



