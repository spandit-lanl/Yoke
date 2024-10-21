"""Test for `datasets/jwl_dataset.py`"""

import unittest
import torch
import yoke.datasets.jwl_dataset as JWL
import numpy as np
import pandas as pd


class TestJWL_Dataset(unittest.TestCase):
    def setUp(self):
        """Set up common variables for the tests."""
        self.rng = slice(0,None)
        self.file = "samples_sand-all.csv"
        # make object to test
        self.model = JWL.CYLEX_pdv2jwl_Dataset(rng=self.rng, file=self.file)
        # load data for testing
        self.data_df = pd.read_csv("samples_sand-all.csv", sep=",", header=0, engine="python")
        # select index of sample to use for testing
        self.itest = 3 
        self.jwl = self.data_df.loc[self.itest,"a":"V0"].values
        self.CJ = self.data_df.loc[self.itest,"dcj":"edet"].values
        self.e = self.data_df.loc[self.itest,"e1":"e7"].values
        self.pdv = self.data_df.loc[self.itest,"t0.1":"t4.5"].values
        self.Nsamples = len(self.data_df)

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, torch.utils.data.Dataset)

    def test_resetCJ(self):
        """Test if model returns correct Dataset length."""
        self.model.resetCJ()
        CJ = self.model.df.loc[self.itest,"dcj":"edet"].values
        np.testing.assert_allclose(CJ,self.CJ)

    def test_len(self):
        """Test if model returns correct Dataset length."""
        self.assertEqual(len(self.model),self.Nsamples)

    def test_getitem(self):
        """Test if model return correct indexed items."""
        input,output = self.model[self.itest]
        np.testing.assert_allclose(input,self.pdv)
        np.testing.assert_allclose(output,self.jwl[0:-1])

    def test_computeCJ(self):
        """Test if model return correct CJ state."""
        CJ = JWL.compute_CJ(*self.jwl)
        # render volume dimensional
        CJ[2] *= self.jwl[-1]
        np.testing.assert_allclose(self.CJ, CJ)

    def test_compute_e_release(self):
        """Test if model return correct availability on release isentrope items."""
        vs = np.linspace(1,7,7)
        ecalc = JWL.compute_e_release(*self.jwl,self.CJ[-1],vs)
        np.testing.assert_allclose(ecalc, self.e)

class TestJWLnorm_Dataset(unittest.TestCase):
    def setUp(self):
        """Set up common variables for the tests."""
        self.rng = slice(0,None)
        self.file = "samples_sand-all.csv"
        # make object to test
        self.model = JWL.CYLEXnorm_pdv2jwl_Dataset(rng=self.rng, file=self.file)
        # load data for testing
        self.data_df = pd.read_csv("samples_sand-all.csv", sep=",", header=0, engine="python")
        # select index of sample to use for testing
        self.itest = 3 
        self.jwl = self.data_df.loc[self.itest,"a":"V0"].values
        self.CJ = self.data_df.loc[self.itest,"dcj":"edet"].values
        self.e = self.data_df.loc[self.itest,"e1":"e7"].values
        self.pdv = self.data_df.loc[self.itest,"t0.1":"t4.5"].values
        self.Nsamples = len(self.data_df)

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, torch.utils.data.Dataset)

    def test_resetCJ(self):
        """Test if model returns correct Dataset length."""
        self.model.resetCJ()
        CJ = self.model.df.loc[self.itest,"dcj":"edet"].values
        np.testing.assert_allclose(CJ,self.CJ)

    def test_len(self):
        """Test if model returns correct Dataset length."""
        self.assertEqual(len(self.model),self.Nsamples)

    def test_getitem(self):
        """Test if model return correct indexed items."""
        input,output = self.model[self.itest]
        np.testing.assert_allclose(input,(self.pdv-self.model.pdvmin)/(self.model.pdvmax-self.model.pdvmin))
        np.testing.assert_allclose(output,(self.jwl[0:-1]-self.model.jwlmins)/(self.model.jwlmaxs-self.model.jwlmins))

    def test_computeCJ(self):
        """Test if model return correct CJ state."""
        CJ = JWL.compute_CJ(*self.jwl)
        # render volume dimensional
        CJ[2] *= self.jwl[-1]
        np.testing.assert_allclose(self.CJ, CJ)

    def test_compute_e_release(self):
        """Test if model return correct availability on release isentrope items."""
        vs = np.linspace(1,7,7)
        ecalc = JWL.compute_e_release(*self.jwl,self.CJ[-1],vs)
        np.testing.assert_allclose(ecalc, self.e)

if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
