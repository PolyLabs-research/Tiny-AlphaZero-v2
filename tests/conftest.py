"""Suite-wide fixtures.

One job: keep process-global state from leaking between tests. Several tests
call ``torch.manual_seed(0)`` to pin a lottery they would otherwise lose, and
without this every later test ran on whatever RNG state the previous one left —
test-order-dependent model inits that no test asked for (REVIEW-2026-10-21 B9,
C10). Forking the global RNGs around each test makes a seed call inside a test
exactly as local as the test believes it is.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        yield
    np.random.set_state(numpy_state)
    torch.set_num_threads(threads)
