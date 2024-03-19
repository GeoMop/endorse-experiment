from endorse.common.memoize2 import ResultCacheFile, memoize
import os


script_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(script_dir, "cache_data")


@memoize(ResultCacheFile(cache_dir))
def fce(a):
    return a


def clear_cache():
    ResultCacheFile(cache_dir).clear()


def read_hash():
    dir = os.path.join(cache_dir, "fce")
    if not os.path.isdir(dir):
        return None
    return os.listdir(dir)[0]


# hash for input
def hi(a):
    clear_cache()
    fce(a)
    return read_hash()


def test_inputs():
    # basic
    assert hi(1) == hi(1)
    assert hi(1) != hi(2)

    # custom classes
    class A:
        def __init__(self, a):
            self.a = a

    a = A(1)
    b = A(2)

    assert hi(a) == hi(a)
    assert hi(a) != hi(b)

    # functions
    a = lambda x: 1
    b = lambda x: 2

    assert hi(a) == hi(a)
    assert hi(a) != hi(b)
