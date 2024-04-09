from typing import *
import os
import sys
import shutil
import pickle
import functools
import hashlib
import cloudpickle


_config = {
    "debug": False
}


def basic_config(**kwargs):
    _config.update(kwargs)


class ResultCacheFile:
    class NoValue:
        pass

    def __init__(self, cache_dir="./cache_data"):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def value(self, hash: bytes, fun_name: str) -> Any:
        dir_path = os.path.join(self._cache_dir, fun_name)
        file_path = os.path.join(dir_path, hash.hex())
        if not os.path.exists(file_path):
            return self.NoValue

        with open(file_path, "rb") as f:
            bin_data = f.read()

        value = pickle.loads(bin_data)
        return value

    def insert(self, hash: bytes, value: Any, fun_name: str):
        bin_data = cloudpickle.dumps(value)

        dir_path = os.path.join(self._cache_dir, fun_name)
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, hash.hex())

        with open(file_path, "wb") as f:
            f.write(bin_data)

    def clear(self):
        for filename in os.listdir(self._cache_dir):
            file_path = os.path.join(self._cache_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)


def memoize(cache):
    def decorator(func):
        cloudpickle.register_pickle_by_value(sys.modules[func.__module__])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            m = hashlib.sha256()
            m.update(cloudpickle.dumps((func, args, kwargs)))
            hash = m.digest()

            fun_name = func.__name__
            value = cache.value(hash, fun_name)
            if value is ResultCacheFile.NoValue:
                result = func(*args, **kwargs)
                data = {
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "result": result
                }
                cache.insert(hash, data, fun_name)
            else:
                result = value["result"]
                if _config["debug"]:
                    new_result = func(*args, **kwargs)
                    if new_result != result:
                        print("Function {}, new result is different from cache result.".format(fun_name))

            return result

        wrapper.__name__ += "_memoize"

        return wrapper
    return decorator
