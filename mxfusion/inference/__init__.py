# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


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
    logger
"""

from .batch_loop import BatchInferenceLoop
from .batch_inference_loop_lbfgs import BatchInferenceLoopLBFGS
from .expectation import ExpectationAlgorithm, ExpectationScoreFunctionAlgorithm
from .forward_sampling import ForwardSampling, VariationalPosteriorForwardSampling, ForwardSamplingAlgorithm
from .grad_based_inference import GradBasedInference, GradTransferInference
from .inference import Inference, TransferInference
from .inference_parameters import InferenceParameters
from .logger import Logger
from .map import MAP
from .meanfield import create_Gaussian_meanfield
from .minibatch_loop import MinibatchInferenceLoop
from .pilco_alg import PILCOAlgorithm
from .prediction import ModulePredictionAlgorithm
from .score_function import ScoreFunctionInference, ScoreFunctionRBInference
from .variational import StochasticVariationalInference
