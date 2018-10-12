import pytest
import mxnet as mx
import numpy as np
import random


@pytest.fixture(scope='session')
def set_seed():
    random.seed(0)
    np.random.seed(0)
    mx.random.seed(0)
