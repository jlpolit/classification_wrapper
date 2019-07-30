from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np


class ClassificationWrapper:
    def __init__(self, model_type="logit", random_state=None):
        # initialize default model objects
        self.model_pipeline = None
        self.random_state = random_state
        self.model_lookup = {'logit': LogisticRegression(l1_ratio=.5,
                                                         solver='saga',
                                                         penalty='elasticnet',
                                                         random_state=random_state),
                             'gbm': GradientBoostingClassifier(random_state=random_state)}
        self.model = self.model_lookup.get(model_type)
        self.param_search_space = None

    @staticmethod
    def get_feature_types(X: pd.DataFrame) -> dict:
        """

        :param X: dataframe
        :return: dictionary indicating which features are numeric and which categorical
        """
        feature_types = X.dtypes.to_dict()
        numerical_features = [x for x in feature_types.keys() if feature_types[x] in [float, int]]
        working_cat_feats = [x for x in feature_types.keys() if feature_types[x] not in [float, int]]
        # check whether any of our categorical features are sneaky numbers
        categorical_features = []
        for cat in working_cat_feats:
            if X[cat].str.isnumeric().sum() == len(X):
                X[cat] = X[cat].astype(float)
                numerical_features.append(cat)
            else:
                categorical_features.append(cat)

        return {'numerical': numerical_features, 'categorical': categorical_features}

    def _build_model_pipeline(self, X: pd.DataFrame) -> None:
        """

        :param X:
        :return:
        """
        # get our feature types
        dtype_lookup = self.get_feature_types(X)
        numerical_features = dtype_lookup['numerical']
        categorical_features = dtype_lookup['categorical']

        # define our pipeline for numerical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        # define our pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # put everything together into one transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])

        self.model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                              ('classifier', self.model)])

    def fit(self, X: pd.DataFrame, y: np.array) -> None:
        """

        :param X: dataframe of predictors
        :param y: numpy array of labels
        :return: N/A. trains a regularized model using default parameters
        """
        self._build_model_pipeline(X)
        self.model_pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.array:
        """

        :param X: data frame of input features
        :return: predicted class labels
        """
        if self.model_pipeline:
            return self.model_pipeline.predict(X)
        else:
            raise NameError("No trained model could be found")

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """

        :param X: data frame of input features
        :return: predicted class probabilities
        """
        if self.model_pipeline:
            return self.model_pipeline.predict_proba(X)
        else:
            raise NameError("No trained model could be found")

    def evaluate(self, X: pd.DataFrame, y: np.array) -> dict:
        """

        :param X: input features in a dataframe
        :param y: ground truth classification labels
        :return: dictionary of log loss and f1 score
        """
        if self.model_pipeline:
            # f1 uses class predictions
            class_predictions = self.model_pipeline.predict(X)
            f1 = f1_score(y, class_predictions)

            # log loss needs predicted probabilities
            predicted_probs = self.model_pipeline.predict_proba(X)
            ll = log_loss(y, predicted_probs)

            return {'f1_score': f1,
                    'logloss': ll}

        else:
            raise NameError("No trained model could be found")

    def _build_search_space(self):
        # to be overwritten in child classes
        raise NotImplementedError

    def tune_parameters(self, X: pd.DataFrame, y: np.array) -> dict:
        """

        note that this will replace the current model pipeline, if any exists

        :param X: input features in a dataframe
        :param y: ground truth classification labels
        :param early_stopping: boolean indicating whether to use early stopping while training
        :return: dictionary of best params, as well as performance
        """
        # not comfortable assuming the previous model pipeline (if any) was built with the same features
        self._build_model_pipeline(X)
        self._build_search_space()

        log_loss_scorer = make_scorer(log_loss)
        f1_scorer = make_scorer(f1_score)

        random_search = RandomizedSearchCV(estimator=self.model_pipeline,
                                           param_distributions= self.param_search_space,
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


