import mock

from mxfusion import Variable
from mxfusion.inference.inference_alg import SamplingAlgorithm
from mxfusion.models import Model
from mxfusion.modules.module import Module


def test_module_clone_algorithms():
    m = Model()
    m.X = Variable()
    m.Y = Variable()

    module = Module([('X', m.X)], [('Y', m.Y)], ['in'], 'out')
    module._module_graph = Model()

    sampling_alg = mock.create_autospec(SamplingAlgorithm)
    sampling_alg.replicate_self.return_value = sampling_alg
    module.attach_draw_samples_algorithms([m.Y], [m.X], sampling_alg)

    cloned_module = module.replicate_self()

    # Check dictionary contains 1 entry
    assert len(cloned_module._draw_samples_algorithms) == 1
    # Check dictionary key is a tuple containing m.X
    assert list(cloned_module._draw_samples_algorithms.keys())[0] == (m.X, )
    # Check dictionary value is a list containing a tuple with 3 entries
    assert isinstance(cloned_module._draw_samples_algorithms[(m.X,)], list)
    assert isinstance(cloned_module._draw_samples_algorithms[(m.X,)][0], tuple)
    assert len(cloned_module._draw_samples_algorithms[(m.X,)][0]) == 3
