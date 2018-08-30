import pytest
import numpy as np


@pytest.fixture(scope='session')
def set_seed():
    np.random.seed(0)
