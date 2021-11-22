#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# insilico/model.py
# v.0.1.1
# Developed in 2021 by Steven Newton <steven.j.newton99@gmail.com>
#
# Contains the `model` class for use with functions.py
#

# stdlib. imports
from os.path import abspath, dirname, join

_DIRECTORY_PATH = abspath(dirname(__file__))
_DATA_PATH = abspath(join(_DIRECTORY_PATH, 'data'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


class Model:
	'''Predict pIC50 based on ChEMBL fingerprint and target protein used.
	Args:
		df: Input dataframe with fingerprints
		var_threshold: Drop columns if variance is less than this (float)
		test_size: Proportion of data to use in test set (float)
		response_var: Response variable, e.g. pIC50 (str)
	'''
	def __init__(self, df, var_threshold=0.15, test_size=0.2, response_var='pIC50'):

		df_model = df.copy() 

		self.y = df_model.pop(response_var) #isolate response variable

		df_model = df_model.iloc[:, 11:] #drop non-fingerprint columns

		#drop low-variance columns and output an array
		self.X = VarianceThreshold(var_threshold).fit_transform(df_model)
		self.test_size = test_size


	def split_data(self):
		'''Splits data for modeling.
		Returns:
			X_train, X_test, y_train, y_test: the split and prepared data (arrays)
		'''
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
															test_size=self.test_size,
															random_state=44)
		print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

		return X_train, X_test, y_train, y_test


	def decision_tree(self, max_depth=50, ccp_alpha=0., save=True):
		'''Do cross-validation, fit, and return metrics for a scikit-learn decision tree.
		Args:
			df: Datafame made from previous class
			max_depth: Max depth of tree (int)
			ccp_alpha: Hyperparameter to control overfitting (float)
			save: To be or not to be, model (bool)
		Returns:
			tree: a decision tree model, can be output to pkl file
			metrics: the train and test support, MAE and R^2 (df)
		'''
		X_train, X_test, y_train, y_test = self.split_data()
		tree = DecisionTreeRegressor(max_depth=max_depth,
										ccp_alpha=ccp_alpha, random_state=44)

		cv_scores = cross_val_score(tree, X_train, y_train, cv=5)
		print("Cross-validation scores:", np.round(cv_scores,3))

		tree.fit(X_train, y_train)
		metrics = self._prediction(tree, X_train, X_test, y_train, y_test)

		if save: pickle.dump(tree, open(join(_DATA_PATH, "tree.pkl"), 'wb'))

		return tree, metrics


	def _prediction(self, mdl, X_train, X_test, y_train, y_test):
		'''Return metrics and residuals for model's predictions.
		Args:
			mdl: A scikit-learn model
			X_train, X_test, y_train, y_test: the split data (arr)
		Returns:
			metrics_dict: Support, MAE and R^2 for train/test (dict)
		'''
		train_predictions = mdl.predict(X_train)
		test_predictions = mdl.predict(X_test)

		metrics_dict = {}
		metrics_dict['support'] = [len(y_train), len(y_test)]
		metrics_dict['mean_absolute_error'] = [round(mean_absolute_error(y_train, train_predictions),5),
												round(mean_absolute_error(y_test, test_predictions),5)]
		metrics_dict['r_squared'] = [round(r2_score(y_train, train_predictions),5),
									round(r2_score(y_test, test_predictions),5)]
		metrics pd.DataFrame(metrics_dict)
		metrics.index = ['train', 'test']
		print(metrics.iloc[1])

		#residual plot
		ax = sns.regplot(x=y_test, y=test_predictions, scatter_kws={'alpha':.4})
		ax.axvline(y_train.median(), color='gray', alpha=.5) #add median line
		ax.text(y_train.median()+.1, 6, 'median', color='gray')
		ax.set_xlabel('Experimental $pIC_{50}$')
		ax.set_ylabel('Predicted $pIC_{50}$')
		ax.set_title('Residual plot')
		IC50_range = (y_test.min()-.2, y_test.max()+.2)
		ax.set_xlim(IC50_range)
		ax.set_ylim(IC50_range)
		plt.savefig(join(_DATA_PATH, "residuals_1.png"))
		plt.show()

		#histogram of errors
		error = test_predictions - y_test
		ax = sns.histplot(error, bins=20)
		ax.set_title('Errors distribution')
		ax.set_xlabel('Prediction Error $pIC_{50}$')
		ax.set_ylabel('count')
		plt.savefig(join(_DATA_PATH, "residuals_2.png"))
		plt.show()

		print('Plots saved in ' + _DATA_PATH)

		return metrics
