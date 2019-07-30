import unittest
import os
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from classwrap.logit_wrapper import LogitWrapper


def get_data():
    df_path_local = '../data/DR_Demo_Lending_Club_reduced.csv'
    if not os.path.isfile(df_path_local):
        df_url = 'https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Lending_Club_reduced.csv'
        df = pd.read_csv(df_url)
        df.to_csv(df_path_local, index=False)
    else:
        df = pd.read_csv(df_path_local)
    return df


class TestClasswrap(unittest.TestCase):
    def setUp(self):
        self.df = get_data()
        self.X_df = self.df.drop(columns=['is_bad'])
        self.y = self.df.is_bad.values
        self.X_train, \
            self.X_test,\
            self.y_train, \
            self.y_test = train_test_split(self.X_df,
                                           self.y,
                                           random_state=0,
                                           stratify=self.y)

    def test_get_feature_types(self):
        feat_type_dict = LogitWrapper.get_feature_types(self.df)
        self.assertIn('annual_inc', feat_type_dict['numerical'])
        self.assertIn('verification_status', feat_type_dict['categorical'])

    def test_fit_predict_evaluate(self):
        logit_model = LogitWrapper(random_state=0)
        logit_model.fit(self.X_train, self.y_train)
        predictions = logit_model.predict(self.X_test)
        predicted_probs = logit_model.predict_proba(self.X_test)
        # assert reproducible results given same train/test split and random seed
        self.assertEqual(sum(predictions), 22)
        self.assertEqual(len(predictions), len(self.y_test))
        # make sure we're getting the values we expect
        self.assertEqual(set(predictions), {0,1})
        # check our predicted probabilities
        self.assertEqual(predicted_probs.shape, (2500, 2))
        self.assertEqual(round(np.mean([x[1] for x in predicted_probs]), 2), 0.13)
        # check for real probabilities
        self.assertEqual(set([sum(x) for x in predicted_probs]), {1.0})
        evaluation_dict = logit_model.evaluate(self.X_test, self.y_test)
        self.assertIn('f1_score', evaluation_dict.keys())
        self.assertIn('logloss', evaluation_dict.keys())
        self.assertEqual(round(evaluation_dict['logloss'], 2), 0.35)
        # to do: K folds
        k_folds_output = logit_model.tune_parameters(self.X_train, self.y_train)
        self.assertIn('classifier__C', k_folds_output.keys())
        self.assertIn('scores', k_folds_output)
        self.assertEqual(round(k_folds_output['scores']['f1_score'], 2), 0.13)
        self.assertEqual(k_folds_output['classifier__C'], 0.1)

    def test_missing_values(self):
        logit_model2 = LogitWrapper(random_state=0)
        # add more nans
        X_miss = copy.deepcopy( self.X_train)
        X_miss['dept_to_income'] = X_miss.debt_to_income.apply(lambda x: x if x > 13 else np.NaN)
        X_miss['home_ownership'] = X_miss.home_ownership.replace('RENT', np.NaN)
        logit_model2.fit(X_miss, self.y_train)
        eval2 = logit_model2.evaluate(self.X_test,self.y_test)
        self.assertEqual( round(eval2['f1_score'], 2), 0.13)

    def test_new_category(self):
        logit_model3 = LogitWrapper(random_state=0)
        # remove some levels
        X_fewer_factor_levels = copy.deepcopy(self.X_train)
        X_fewer_factor_levels['purpose_cat'] = X_fewer_factor_levels.purpose_cat.replace('home improvement', 'car')
        X_fewer_factor_levels['home_ownership'] = X_fewer_factor_levels.home_ownership.replace('RENT', 'OTHER')
        logit_model3.fit(X_fewer_factor_levels, self.y_train)
        self.assertEqual(round(logit_model3.evaluate(self.X_test, self.y_test)['logloss'],1), 0.3)


if __name__ == '__main__':
    unittest.main()

