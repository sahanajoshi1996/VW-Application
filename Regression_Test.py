import imp


import unittest
from Trainer import NewTrainer

class Regression_Tests(unittest.TestCase):
    def setUp(self):
        self.trainer = NewTrainer()
    
    def test_train_input_shape(self):
        x = [[2., 3., 4., 1.], [1., 7., 3., 5.3], [8.1, 7.2, 4.1, 3.4]]
        y = [1., 2.4]
        self.trainer.train(x, y)
        self.assertRaises(Exception, 'Raises exception that input x and y have differnt lengths')

    def test_train_shape_trained_parameters(self):
        x = [[2.], [1], [8.], [3.4]]
        y = [2., 1, 8., 3.4]
        self.trainer.train(x, y)
        params = self.trainer.params
        expt_params = [1., 0]
        self.assertEqual(len(params), len(expt_params), "Shape of model paramaters is incorrect")

    def test_train_trained_parameters(self):
        x = [[2.], [1], [8.], [3.4]]
        y = [2., 1, 8., 3.4]
        self.trainer.train(x, y)
        params = self.trainer.params
        expt_params = [1., 0]
        self.assertEqual(params, expt_params, "Model parameters incorrect, training may be incorrect")

    def test_predict_return(self):
        x = [[2.], [1], [8.], [3.4]]
        y = [4., 2, 16., 6.8]
        self.trainer.train(x, y)
        eval_in = [5.6]
        eval_out = self.trainer.predict(eval_in)
        eval_out = eval_out[0]

        self.assertEqual(float, type(eval_out), "Function did not return float datatype")

    def test_predict(self):
        x = [[2.], [1], [8.], [3.4]]
        y = [4., 2, 16., 6.8]
        self.trainer.train(x, y)
        eval_in = [5.6]
        eval_out = self.trainer.predict(eval_in)
        expt_eval_out = [11.2]
        self.assertEqual(eval_out, expt_eval_out, "Predicted output incorrect")
