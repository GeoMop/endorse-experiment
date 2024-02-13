from endorse.common.memoize2 import ResultCacheFile, memoize
import os
import time


script_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(script_dir, "cache_data")

def test_cache():
    cache = ResultCacheFile(cache_dir)
    cache.clear()

    hash = b"abc"
    value = (1, "hello")
    fun_name = "name"

    cache.insert(hash, value, fun_name)

    from_cache = cache.value(hash, fun_name)

    assert from_cache == value


@memoize(ResultCacheFile(cache_dir))
def func(a:int):
    print("\nCompute func.")
    time.sleep(2)
    return a * a


def test_memoization():
    cache = ResultCacheFile(cache_dir)
    cache.clear()

    f1 = func(2)
    f2 = func(2)
    f3 = func(f1 + f2)
