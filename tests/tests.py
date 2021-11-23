#to run: python -m unittest -v tests/tests.py
import unittest
from insilico import target_search, process_target_data, ModelChembl

class ChemblTestCase(unittest.TestCase):

    def setUp(self):
        """Search chembl_db for a target organism, P. knowlesi"""

        res = target_search('P. knowlesi')
        self.assertGreater(len(res), 0)
        # assign top result to chembl_id variable
        self.chembl_id = res.target_chembl_id[0]

    def test(self):
        """Test processing data and modeling class"""

        df = process_target_data(self.chembl_id)

        #test Model class using a decision tree
        mdl = ModelChembl(df, var_threshold=.1, test_size=.3)
        tree, preds = mdl.tree(max_depth=5, ccp_alpha=0)
        metrics = mdl.evaluate(preds)
        X_train, X_test, y_train, y_test = mdl.get_data()
        
        self.assertGreater(metrics['support'], 0)
        self.assertGreater(len(preds), 0)

if __name__ == '__main__':
    unittest.main()