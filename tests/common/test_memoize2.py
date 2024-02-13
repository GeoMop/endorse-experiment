from endorse.common.memoize2 import ResultCacheFile
import os


script_dir = os.path.dirname(os.path.realpath(__file__))


def test_cache():
    cache = ResultCacheFile(os.path.join(script_dir, "cache_data"))

    hash = b"abc"
    value = (1, "hello")
    fun_name = "name"

    cache.insert(hash, value, fun_name)

    from_cache = cache.value(hash, fun_name)

    assert from_cache == value

    cache.clear()
