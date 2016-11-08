"""Test for memory leaks in pescador streamers.

Call
----
py.test -vs test_memleak.py


Expected versions
-----------------
NumPy: 1.11.2
Pandas: 0.18.0
Pescador: 0.1.3
"""

import pytest

import numpy as np
import os
import pandas as pd
import pescador
import tempfile
import shutil
import uuid


NUM_ARRAYS = 20
TSIZE = (4096, 128)
VERBOSE = True


@pytest.fixture(scope='module')
def workspace(request):
    """Returns a path to a temporary directory for writing data."""
    test_workspace = tempfile.mkdtemp()

    def fin():
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)

    request.addfinalizer(fin)
    return test_workspace


@pytest.fixture(scope='module')
def index(workspace):
    recs = []
    index = []
    for n in range(NUM_ARRAYS):
        if (n % 100) == 0 and VERBOSE:
            print("{} / {}".format(n, NUM_ARRAYS))
        key = str(uuid.uuid4())
        data = dict(x=np.random.normal(0, 1, size=TSIZE),
                    y=np.random.uniform(0, 1, size=TSIZE[:1]))
        fout = os.path.join(workspace, "{}.npz".format(key))
        np.savez(fout, **data)
        recs += [dict(filename=fout)]
        index += [key]

    return pd.DataFrame.from_records(recs, index=index)


def test_setup(index):
    print("NumPy: {}".format(np.version.version))
    print("Pandas: {}".format(pd.__version__))
    print("Pescador: {}".format(pescador.version.version))


def test_read_all(index):
    """Will consume a teeny bit of memory"""
    print("Loading all data....")
    data = []
    for n, fn in enumerate(index.filename):
        if (len(data) % 100) == 0 and data and VERBOSE:
            print("{} / {}: {}".format(n, len(index), data[-1].shape))
        data += [np.load(fn)['x']]


def npz_sampler(row):
    """Draw random rows from a record.

    Parameters
    ----------
    row : pd.Series
        Row containing a pointer to an npz file under "filename".

    Yields
    ------
    sample : dict
        An np.ndarray, under 'x'.
    """
    with np.load(row.filename) as data:
        x = data['x']

    while True:
        i = np.random.randint(len(x))
        yield dict(x=x[i:i + 1])


def sample_streamer(index, working_size, lam, sample_func):
    seed_pool = [pescador.Streamer(sample_func, row)
                 for idx, row in index.iterrows()]

    return pescador.mux(seed_pool, n_samples=None, k=working_size,
                        lam=lam, with_replacement=True)


def test_stream_many(index):
    """Should consume minimal memory, but will crash after 5k samples."""
    n_samples = 50000
    print("Streaming data")
    stream = sample_streamer(index, working_size=20,
                             lam=5, sample_func=npz_sampler)

    for n, sample in enumerate(stream):
        if (n % 100) == 0 and n and VERBOSE:
            print("{} / {}: {}".format(n, n_samples, sample['x'].shape))
        elif n > n_samples:
            break
