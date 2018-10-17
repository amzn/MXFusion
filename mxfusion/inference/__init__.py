"""This module contains inference related methods and classes.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    batch_loop
    forward_sampling
    grad_based_inference
    grad_loop
    inference_alg
    inference_parameters
    inference
    map
    meanfield
    minibatch_loop
    variational
"""


from .map import MAP
from .batch_loop import BatchInferenceLoop
from .inference import Inference, TransferInference
from .minibatch_loop import MinibatchInferenceLoop
from .meanfield import create_Gaussian_meanfield
from .forward_sampling import ForwardSampling, VariationalPosteriorForwardSampling
from .grad_based_inference import GradBasedInference
from .variational import StochasticVariationalInference
from .inference_parameters import InferenceParameters
from .score_function import ScoreFunctionInference, ScoreFunctionRBInference
from .prediction import ModulePredictionAlgorithm
