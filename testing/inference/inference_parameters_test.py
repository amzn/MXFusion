import unittest
import mxnet as mx
from mxfusion.components import Variable
from mxfusion.inference import InferenceParameters


class InferenceParametersTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.inference.InferenceParameters methods.
    """

    def remove_saved_files(self, prefix):
        import os, glob
        for filename in glob.glob(prefix+"*"):
            os.remove(filename)

    def test_save_reload_constants(self):
        constants = {Variable(): 5, 'uuid': mx.nd.array([1])}
        ip = InferenceParameters(constants=constants)
        ip.save(prefix="constants_test")
        # assert the file is there

        ip2 = InferenceParameters.load_parameters(
            mxnet_constants_file='constants_test_mxnet_constants.json',
            variable_constants_file='constants_test_variable_constants.json')
        print(ip.constants)
        print(ip2.constants)
        assert ip.constants == ip2.constants

        self.remove_saved_files("constants_test")
