#to run: python -m unittest -v tests/tests.py
import unittest
from insilico import target_search, process_target_data, Model

class ChemblTestCase(unittest.TestCase):

    def setUp(self):
        """Search chembl_db for a target organism, P. knowlesi"""

        res = target_search('P. knowlesi')
        self.assertGreater(len(res), 0)

        # assign top result to chembl_id variable
        self.chembl_id = res.target_chembl_id[0]

    def test_functions(self):
        """Test Process data, visuals and modeling class"""

        df = process_target_data(self.chembl_id, plots=False)

        #test Model class using a decision tree
        mdl = Model(df, var_threshold=0, test_size=.1)

        X_train, X_test, y_train, y_test = mdl.split_data()

        model, metrics = mdl.decision_tree(max_depth=3, ccp_alpha=0, save=True)
        
        self.assertGreater(metrics['support'][0], 0)

if __name__ == '__main__':
    unittest.main()