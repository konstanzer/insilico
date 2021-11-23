#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# insilico/model.py
# v.0.1.2
# Developed in 2021 by Steven Newton <steven.j.newton99@gmail.com>
#
# Contains the `ModelChembl` class for use with functions.py
#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class ModelChembl:
	'''Predict pIC50 based on ChEMBL fingerprint and target protein used.
	Args:
		df: Input dataframe with fingerprints
		var_threshold: Drop columns if variance is less than this (float)
		test_size: Proportion of data to use in test set (float)
	'''
	def __init__(self, df, var_threshold=0.15, test_size=0.2):

		df_model = df.copy()
		y = df_model.pop('pIC50') #isolate response variable
		df_model = df_model.iloc[:, 11:] #drop non-fingerprint columns
		X = VarianceThreshold(var_threshold).fit_transform(df_model)
		
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
													X, y, test_size=test_size, random_state=44)

	def tree(self, max_depth=50, ccp_alpha=0.):
		'''Do cross-validation, fit, and return metrics for a scikit-learn decision tree.
		Args:
			df: Datafame made from previous class
			max_depth: Max depth of tree (int)
			ccp_alpha: Hyperparameter to control overfitting (float)
		Returns:
			tree: a decision tree model, can be output to pkl file
			(predictions): the test set predictions
		'''
		tree = DecisionTreeRegressor(max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=44)
		scores = cross_val_score(tree, self.X_train, self.y_train, cv=5) #R^2 scoring
		print("5-Fold cross-validation results:")
		print("%0.2f r_squared with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
		tree.fit(self.X_train, self.y_train)

		return tree, tree.predict(self.X_test)

	def evaluate(self, predictions):
		'''Return metrics and print residual plots for a model.
		Args:
			predictions: Predictions from the fitted model (arr)
		Returns:
			metrics: Support, MAE and R^2 for test data (dict)
		'''
		#residual plot
		ax = sns.regplot(x=self.y_test, y=predictions, scatter_kws={'alpha':.4})
		ax.axvline(self.y_train.median(), color='gray', alpha=.5) #add median line
		ax.text(self.y_train.median()+.1, 6, 'median', color='gray')
		ax.set_xlabel('Experimental $pIC_{50}$')
		ax.set_ylabel('Predicted $pIC_{50}$')
		ax.set_title('Residual plot')
		IC50_range = (self.y_test.min()-.2, self.y_test.max()+.2)
		ax.set_xlim(IC50_range)
		ax.set_ylim(IC50_range)
		plt.show()
		#histogram of errors
		ax = sns.histplot(predictions - self.y_test, bins=20)
		ax.set_title('Errors distribution')
		ax.set_xlabel('Prediction Error $pIC_{50}$')
		ax.set_ylabel('')
		plt.show()

		metrics = dict(support=len(self.y_test),
						mean_absolute_error=round(mean_absolute_error(self.y_test, predictions),5),
						r_squared=round(r2_score(self.y_test, predictions),5))

		return metrics

	def get_data(self):
		'''For use with models outside of class.'''
		return self.X_train, self.X_test, self.y_train, self.y_test