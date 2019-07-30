from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
from scipy.stats import uniform

from .base_wrapper import ClassificationWrapper


class LogitWrapper(ClassificationWrapper):

    def tune_parameters(self, X: pd.DataFrame, y: np.array) -> dict:
        """

        note that this will replace the current model pipeline, if any exists

        :param X: input features in a dataframe
        :param y: ground truth classification labels
        :return: dictionary of best params, as well as performance
        """
        # not comfortable assuming the previous model pipeline (if any) was built with the same features
        self._build_model_pipeline(X)
        param_search_space = {
            'preprocessor__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
            'classifier__C': [0.1, 1.0, 10, 100],
            'classifier__l1_ratio': uniform(0, 1)
        }

        log_loss_scorer = make_scorer(log_loss)
        f1_scorer = make_scorer(f1_score)

        random_search = RandomizedSearchCV(estimator=self.model_pipeline,
                                           param_distributions= param_search_space,
                                           scoring= {'f1': f1_scorer,
                                                     'log_loss': log_loss_scorer},
                                           random_state=self.random_state,
                                           refit='log_loss')
        random_search.fit(X, y)
        # update our pipeline
        self.model_pipeline = random_search.best_estimator_
        # get our performance and winning params
        params = random_search.best_params_
        best_model = pd.DataFrame(random_search.cv_results_).iloc[random_search.best_index_]
        f1 = best_model['mean_test_f1']
        ll = best_model['mean_test_log_loss']
        params['scores'] = {'f1_score': f1, 'logloss': ll}
        return params





