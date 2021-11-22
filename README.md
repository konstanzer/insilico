# insilico: A Python package to process & model ChEMBL data.

[![PyPI version](https://badge.fury.io/py/insilico.svg)](https://badge.fury.io/py/insilico)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

ChEMBL is a manually curated chemical database of bioactive molecules with drug-like properties. It is maintained by the European Bioinformatics Institute (EBI), of the European Molecular Biology Laboratory (EMBL) based in Hinxton, UK.

`insilico` helps drug researchers find promising compounds for drug discovery. It preprocesses ChEMBL molecular data and outputs Lapinski's descriptors and chemical fingerprints using popular bioinformatic libraries. Additionally, this package can be used to make a decision tree model that predicts drug efficacy.

### About the package name

The term *in silico* is a neologism used to mean pharmacology hypothesis development & testing performed via computer (silicon), and is related to the more commonly known biological terms *in vivo* ("within the living") and *in vitro* ("within the glass".)

## Installation

Installation via pip:

```
$ pip install insilico
```

Installation via cloned repository:

```
$ git clone https://github.com/konstanzer/insilico
$ cd insilico
$ python setup.py install
```

### Python dependencies

For preprocessing, `rdkit-pypi`, `padelpy`, and `chembl_webresource_client` and for modeling, `sklearn` and `seaborn`

## Basic Usage

`insilico` offers two functions: one to search the ChEMBL database and a second to output preprocessed ChEMBL data based on the molecular ID. Using the chemical fingerprint from this output, the `Model` class creates a decision tree and outputs residual plots and metrics.

The function `process_target_data` saves the chemical fingerprint and, optionally, molecular descriptor plots to a data folder if `plots=True`.

When declaring the model class, you may specify a test set size and a variance threshold, which sets the minimum variance allowed for each column. This optional step may eliminate hundreds of features unhelpful for modeling. When calling the `decision_tree` function, optionally specify max tree depth and cost-complexity alpha, hyperparameters to control overfitting. If `save=True`, the model is saved to the data folder.

```python
from insilico import target_search, process_target_data, Model

# return search results for 'P. falciparum D6'
result = target_search('P. falciparum')

# returns a dataframe of molecular data for CHEMBL2367107 (P. falciparum D6)
df = process_target_data('CHEMBL2367107')

model = Model(test_size=0.2, var_threshold=0.15)

# returns a decision tree and metrics (R^2 and MAE) & saves residual plot
tree, metrics = model.decision_tree(df, max_depth=50, ccp_alpha=0.)

# returns split data for use in other models
X_train, X_test, y_train, y_test = model.split_data()
```

### Advanced option: Use optional 'fp' parameter to specify fingerprinter

Valid fingerprinters are "PubchemFingerprinter" (default), "ExtendedFingerprinter", "EStateFingerprinter", "GraphOnlyFingerprinter", "MACCSFingerprinter", "SubstructureFingerprinter", "SubstructureFingerprintCount", "KlekotaRothFingerprinter", "KlekotaRothFingerprintCount", "AtomPairs2DFingerprinter", and "AtomPairs2DFingerprintCount".

```python
df = process_target_data('CHEMBL2367107', plots=False, fp='SubstructureFingerprinter')
```

## Contributing, Reporting Issues & Support

Make a pull request if you'd like to contribute to `insilico`. Contributions should include tests for new features added and documentation. File an issue to report problems with the software or feature requests. Include information such as error messages, your OS/environment and Python version.

Questions may be sent to Steven Newton (steven.j.newton99@gmail.com).

## References

[Bioinformatics Project from Scratch: Drug Discovery](https://www.youtube.com/watch?v=plVLRashaA8&list=PLtqF5YXg7GLlQJUv9XJ3RWdd5VYGwBHrP) by Chanin Nantasenamat

