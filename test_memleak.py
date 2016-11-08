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
import psutil
import tempfile
import shutil
import uuid


NUM_ARRAYS = 20
TSIZE = (4096, 128)
VERBOSE = True


# Fixtures
# --------
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
    if VERBOSE:
        print()
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


# Functions
# ---------
def data_reader(index):
    for fn in index.filename:
        yield dict(x=np.load(fn)['x'])


def freeing_sampler_a(row):
    """writeme"""
    with np.load(row.filename) as data:
        x = data['x']

    while True:
        i = np.random.randint(len(x))
        yield dict(x=np.array(x[i:i + 1]))


def freeing_sampler_b(row):
    """writeme"""
    x = np.load(row.filename)['x']
    while True:
        i = np.random.randint(len(x))
        yield dict(x=np.array(x[i:i + 1]))


def freeing_sampler_c(row):
    """writeme"""
    with np.load(row.filename) as data:
        x = data['x']

    x_obs = np.zeros_like(x[:1])
    while True:
        i = np.random.randint(len(x))
        np.copyto(x_obs, x[i:i + 1])
        yield dict(x=x_obs)


def leaky_sampler_a(row):
    """writeme"""
    with np.load(row.filename) as data:
        x = data['x']

    while True:
        i = np.random.randint(len(x))
        yield dict(x=x[i:i + 1])


def leaky_sampler_b(row):
    """writeme"""
    x = np.array(np.load(row.filename)['x'])
    while True:
        i = np.random.randint(len(x))
        yield dict(x=x[i:i + 1])


def sample_streamer(index, working_size, lam, sample_func):
    seed_pool = [pescador.Streamer(sample_func, row)
                 for idx, row in index.iterrows()]

    return pescador.mux(seed_pool, n_samples=None, k=working_size,
                        lam=lam, with_replacement=True)


def is_out_of_memory(gb_free=0.5):
    stats = psutil.virtual_memory()
    return (stats.free / 1024 / 1024 / 1024) <= gb_free


def consume(stream, max_samples, gb_free, should_fail, print_freq=100):
    if VERBOSE:
        print()

    outputs = []
    for n, sample in enumerate(stream):
        x = sample['x']
        outputs += [x]
        mem_maxed = is_out_of_memory(gb_free)

        if (n % print_freq) == 0 and n and VERBOSE:
            print("{} / {}: {}".format(n, max_samples, x.shape))

        if n > max_samples or mem_maxed:
            break

    assert mem_maxed == should_fail


# Tests
# -----
def test_setup(index):
    print("NumPy: {}".format(np.version.version))
    print("Pandas: {}".format(pd.__version__))
    print("Pescador: {}".format(pescador.version.version))


def test_is_out_of_memory():
    assert not is_out_of_memory(2.0)


@pytest.mark.skipif(is_out_of_memory(2.0), reason="Requires more free memory.")
def test_read_all(index):
    """Will consume less than 1GB of memory"""
    consume(data_reader(index), len(index), 1.0, False, print_freq=1)


@pytest.mark.skipif(is_out_of_memory(2.0), reason="Requires more free memory.")
def test_sample_streamer_freeing_a(index):
    """Should consume minimal memory."""
    stream = sample_streamer(index, working_size=20, lam=5,
                             sample_func=freeing_sampler_a)
    consume(stream, 10000, 1.0, False, print_freq=500)


@pytest.mark.skipif(is_out_of_memory(2.0), reason="Requires more free memory.")
def test_sample_streamer_freeing_b(index):
    """Should consume minimal memory."""
    stream = sample_streamer(index, working_size=20, lam=5,
                             sample_func=freeing_sampler_b)
    consume(stream, 10000, 1.0, False, print_freq=500)


@pytest.mark.skipif(is_out_of_memory(2.0), reason="Requires more free memory.")
def test_sample_streamer_freeing_c(index):
    """Should consume minimal memory."""
    stream = sample_streamer(index, working_size=20, lam=5,
                             sample_func=freeing_sampler_c)
    consume(stream, 10000, 1.0, False, print_freq=500)


@pytest.mark.skipif(is_out_of_memory(2.0), reason="Requires more free memory.")
def test_sample_streamer_leaky(index):
    """Should consume minimal memory, but will crash after 5k samples."""
    stream = sample_streamer(index, working_size=20, lam=5,
                             sample_func=leaky_sampler_a)
    consume(stream, 10000, 1.0, True, print_freq=100)
