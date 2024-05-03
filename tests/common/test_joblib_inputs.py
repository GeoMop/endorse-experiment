import os
import time
from joblib import Memory


script_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(script_dir, "cache_data")

memory = Memory(cache_dir, verbose=0)


@memory.cache
def fce(a):
    time.sleep(0.1)
    return a


# from cache
def c(a):
    t = time.time()
    fce(a)
    return time.time() - t > 0.05


def test_inputs():
    # basic
    memory.clear()
    c(1)
    assert not c(1)
    assert c(2)

    # custom classes
    class A:
        def __init__(self, a):
            self.a = a

    a = A(1)
    b = A(2)

    memory.clear()
    # c(a)
    # assert not c(a)
    # assert c(b)

    # functions
    a = lambda x: 1
    b = lambda x: 2

    memory.clear()
    # c(a)
    # assert not c(a)
    # assert c(b)
