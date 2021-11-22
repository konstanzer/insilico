#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# insilico/functions.py
# v.0.1.1
# Developed in 2021 by Steven Newton <steven.j.newton99@gmail.com>
#
# Contains functions to process and explore ChEMBL molecular data
#

# stdlib. imports
from os.path import abspath, dirname, join

_DIRECTORY_PATH = abspath(dirname(__file__))
_DATA_PATH = abspath(join(_DIRECTORY_PATH, 'data'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from padelpy import padeldescriptor


def target_search(drug_target):
	'''Search ChEMBL for target protein and output search results.
	Args:
		drug_target: any search phrase related to the drug target (str)
	Returns:
		Dataframe of ChEMBL target search results
	'''
	results = new_client.target.search(drug_target)

	return pd.DataFrame(results).drop(columns=['cross_references'])


def process_target_data(chembl_id, plots=False, fp="PubchemFingerprinter",
						standard_type='IC50'):
	'''Query ChEMBL database for the given CHEMBL molecule ID, which can be found
		with the search function, and process the target data.
	Args:
		chembl_id: The molecule ID, e.g. CHEMBL364 (str)
		plots: Display and save plots of descriptors (bool)
		fp: Specify fingerprinter file to use, not including XML extension (str)
		standard_type: The measure of drug efficacy, defaults to IC50 (str)
	Returns:
		df: Dataframe with molecular descriptors & chemical fingerprint
	'''
	df = _query_chembl(chembl_id, standard_type)
	df = _bioactivity_class(df) #add column
	df = _lipinski(df) #add descriptors
	df = _pIC50(df) #add column

	if plots: _plot_descriptors(df)

	fps = _compute_fingerprints(chembl_id, df, fp)
	df = df.merge(fps, left_on='molecule_chembl_id', right_on='Name')

	return df


def _query_chembl(chembl_id, standard_type):
	'''Query ChEMBL database and return data for the given target.
	Args:
		chembl_id: The molecule ID, e.g. CHEMBL364 (str)
		standard_type: Filter for measure of drug efficacy (str)
	Returns:
		Subset of query results, minus rows with missing standard_type (df)
	'''
	result = new_client.activity.filter(target_chembl_id=chembl_id)\
								.filter(standard_type=standard_type)
	df = pd.DataFrame(result).dropna(subset=['value']).reset_index(drop=True)
	print(f'Query returned {len(df)} molecules for ' + chembl_id)

	return df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]


def _bioactivity_class(df):
	'''Divide compounds into potency classes.
	Args:
		df: The dataframe returned from querying ChEMBL
	Returns:
		df: The dataframe with added potency class column
	'''
	df.standard_value = df.standard_value.astype('float64')
	class_names = []

	for i in df.standard_value:
		if i >= 10000:
			class_names.append('inactive')
		elif i <= 1000:
			class_names.append('active')
		else:
			class_names.append('intermediate')

	df['bioactivity_class'] = class_names

	return df


def _lipinski(df):
	'''Using SMILES notation, returns the four parameters described by
		Lipinski's Rule of Five in dataframe.
	Args:
		df: The dataframe returned from querying ChEMBL
	Returns:
		df: The dataframe with added Lipinski descriptors
	'''
	smiles = df.canonical_smiles
	moldata = [Chem.MolFromSmiles(elem) for elem in smiles]
	descriptors = pd.DataFrame(data=np.zeros((len(moldata), 4)),
					columns=['MW', 'LogP', 'NumHDonors', 'NumHAcceptors'])

	for ix, mol in enumerate(moldata):
		descriptors.loc[ix] = [Descriptors.MolWt(mol),Descriptors.MolLogP(mol),
								Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)]

	df = pd.concat([df, descriptors], axis=1)

	return df


def _pIC50(df):
	'''Convert IC50 to pIC50 scale and capping input at 100M,
	which would give negative values after negative logarithm.
	Args:
		df: The dataframe without pIC50 class column
	Returns:
		df: The dataframe with added pIC50 class column
	'''
	pIC50 = []

	for ic in df.standard_value:
		ic = min(ic, 1e8) #caps values
		molar = ic * 1e-9 #converts nanomolar to molar
		value = round(-np.log10(molar), 5) #uses 6 significant digits
		pIC50.append(value)
	
	df['pIC50'] = pIC50

	return df


def _compute_fingerprints(chembl_id, df, fp):
	'''Computes and outputs the molecule's binary substructure fingerprint.
	Args:
		chembl_id: The molecule ID, e.g. CHEMBL364 (str)
		df: Dataframe containing SMILES notation
		fp: The type of fingerprinter (str), from XML filenames
	Returns:
		Dataframe of chemical fingerprint.
	'''
	#input file with SMILES notation
	input_file = join(_DATA_PATH, "molecule.smi")
	df_selection = df[['canonical_smiles','molecule_chembl_id']]
	df_selection.to_csv(input_file, sep='\t', index=False, header=False)

	#output file for fingerprint results
	output_file = join(_DATA_PATH, chembl_id+"_fingerprint.csv")

	#path to fingerprinter XML file (obtained at github.com/dataprofessor/padel)
	descriptortype = join(_DIRECTORY_PATH, "fingerprints_xml", fp+".xml")
	
	print('Computing fingerprints (takes several minutes if molecule count 1000+)...')
	padeldescriptor(mol_dir=input_file,
					d_file=output_file,
					descriptortypes=descriptortype,
					detectaromaticity=True, 
					standardizenitro=True, 
					standardizetautomers=True,
					threads=2, 
					removesalt=True,
					log=False,
					fingerprints=True)
	print('Success!')

	return pd.read_csv(output_file) #load fingerprint from file


def _plot_descriptors(df):
	'''Drop intermediate class and plot chemical space &
		boxplots of Lapinski's descriptors for active and inactive molecules.
	'''
	df_exp = df.copy()
	df_exp = df_exp[df_exp.bioactivity_class != 'intermediate']

	plt.figure(figsize=(10, 5))
	sns.scatterplot(x='MW', y='LogP', data=df_exp, palette = 'Set2',
					hue='bioactivity_class', size='pIC50', alpha=.7)
	plt.legend(bbox_to_anchor=(1.5, 1))
	plt.title('Chemical spaces')
	plt.savefig(join(_DATA_PATH, "chem_space.png"))
	plt.show()

	for descriptor in ['pIC50', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']:
		plt.figure(figsize=(6, 1.5))
		sns.boxplot(y='bioactivity_class', x=descriptor, palette='Set2', data=df_exp)
		plt.xlabel('')
		plt.savefig(join(_DATA_PATH, descriptor+".png"))
		plt.show()

	print('Plots saved in ' + _DATA_PATH)
