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
"""


from .map import MAP
from .batch_loop import BatchInferenceLoop
from .inference import Inference, TransferInference
from .minibatch_loop import MinibatchInferenceLoop
from .meanfield import create_Gaussian_meanfield
from .forward_sampling import ForwardSampling, VariationalPosteriorForwardSampling, ForwardSamplingAlgorithm
from .grad_based_inference import GradBasedInference
from .variational import StochasticVariationalInference
from .inference_parameters import InferenceParameters
from .score_function import ScoreFunctionInference, ScoreFunctionRBInference
from .expectation import ExpectationAlgorithm, ExpectationScoreFunctionAlgorithm
from .prediction import ModulePredictionAlgorithm
