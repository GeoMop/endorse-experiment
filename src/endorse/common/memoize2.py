from typing import *
import os
import sys
import shutil
import pickle
import functools
import hashlib
import cloudpickle
import io
import pickletools

"""
TODO:
- comments
- remember function hashing first time it is called
- move code and tests together with simple joblib usage in 'endorse'
  into bgem.fn module
- join configuration with joblib
- add timing and correct logging
- 
"""

# memoization configuration
_config = {
    # debug - do not reuse cached values, but repeat the calculation and check that the results match.
    "debug": False
}


def configure(**kwargs):
    """
    Set memoization options. Global variable.
    """
    _config.update(kwargs)


class ResultCacheFile:
    """
    A simple result cache using separate files for the pickled objects.
    """
    class NoValue:
        pass

    def __init__(self, cache_dir="./cache_data"):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def value(self, hash: bytes, fun_name: str) -> Any:
        """
        Ask for the stored call of the function 'fun_name'.
        The function with its arguments is represented by its 'hash'.
        Return NoValue class if result not found.
        """
        dir_path = os.path.join(self._cache_dir, fun_name)
        file_path = os.path.join(dir_path, hash.hex())
        if not os.path.exists(file_path):
            return self.NoValue

        with open(file_path, "rb") as f:
            bin_data = f.read()

        value = pickle.loads(bin_data)
        return value

    def insert(self, call_hash: bytes, value: Any, fun_name: str):
        """
        Store the return value of the call to the function 'fun_name'.
        The function with its arguments is represented by its 'hash'.
        """
        bin_data = cloudpickle.dumps(value)

        dir_path = os.path.join(self._cache_dir, fun_name)
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, call_hash.hex())

        with open(file_path, "wb") as f:
            f.write(bin_data)

    def clear(self):
        """
        Empty the cache.
        """
        for filename in os.listdir(self._cache_dir):
            file_path = os.path.join(self._cache_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)


def memoize(cache):
    """
    Decorator to memoize call of a function hashing its arguments and the function code itself.
    The function code is inspected and referenced other functions are hashed if
    a) they are from the same module
    b) they are decorated by memoize itself
    :param cache: A caching class providing 'insert' and 'value' methods.
    Simple implementation is provided by ResultCacheFile. Storing pickled results in separate files.
    Not suitable for many small results.
    :return: decorated function
    """
    def decorator(func):
        cloudpickle.register_pickle_by_value(sys.modules[func.__module__])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            m = hashlib.sha256()
            pickled_fn_call = cloudpickle.dumps((func, args, kwargs))
            m.update(pickle_remove_duplicit(pickled_fn_call))
            hash_fn_call = m.digest()

            fun_name = func.__name__
            value = cache.value(hash_fn_call, fun_name)
            if value is ResultCacheFile.NoValue:
                result = func(*args, **kwargs)
                data = {
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "result": result
                }
                cache.insert(hash_fn_call, data, fun_name)
            else:
                result = value["result"]
                if _config["debug"]:
                    new_result = func(*args, **kwargs)
                    if new_result != result:
                        # Detection of faulty hashing.
                        # TODO: better reporting
                        print("Function {}, new result is different from cache result.".format(fun_name))

            return result

        wrapper.__name__ += "_memoize"

        return wrapper
    return decorator


def pickle_remove_duplicit(pkl):
    """
    Optimize a pickle microcode by removing unused PUT opcodes and duplicated unicode strings.
    This is critical to avoid spurious hash differences.
    """
    put = 'PUT'
    get = 'GET'
    oldids = set()  # set of all PUT ids
    newids = {}     # set of ids used by a GET opcode; later used to map used ids.
    opcodes = []    # (op, idx) or (pos, end_pos)
    proto = 0
    protoheader = b''
    strings = {}
    memo_map = {}
    last_opcode_name = ""
    last_arg = None

    # Generate all opcodes and store positions to calculate end_pos
    ops = list(pickletools.genops(pkl))
    for i, (opcode, arg, pos) in enumerate(ops):
        # Determine end_pos by looking at the position of the next opcode
        end_pos = ops[i + 1][2] if i + 1 < len(ops) else len(pkl)

        if 'PUT' in opcode.name:
            assert opcode.name in ('PUT', 'BINPUT'), f"{opcode.name}"
            oldids.add(arg)
            opcodes.append((put, arg))
        elif opcode.name == 'MEMOIZE':
            idx = len(oldids)

            # Inserted into optimize
            if 'BINUNICODE' in last_opcode_name:
                assert last_opcode_name in ('BINUNICODE', 'SHORT_BINUNICODE'), f"{last_opcode_name}"
                if last_arg in strings:
                    opcodes.pop()
                    strid = strings[last_arg]
                    newids[strid] = None
                    opcodes.append((get, strid))
                    memo_map[idx] = strid
                else:
                    strings[last_arg] = idx

            oldids.add(idx)
            opcodes.append((put, idx))
        elif 'FRAME' in opcode.name:
            assert 'FRAME' == opcode.name, f"{opcode.name}"
            pass
        elif 'GET' in opcode.name:
            assert opcode.name in ('GET', 'BINGET'), f"{opcode.name}"
            if opcode.proto > proto:
                proto = opcode.proto

            # inserted into optimize
            if arg in memo_map:
                arg = memo_map[arg]

            newids[arg] = None
            opcodes.append((get, arg))
        elif opcode.name == 'PROTO':
            if arg > proto:
                proto = arg
            if pos == 0:
                protoheader = pkl[pos:end_pos]
            else:
                opcodes.append((pos, end_pos))
        else:
            opcodes.append((pos, end_pos))
        last_opcode_name = opcode.name
        last_arg = arg
    del oldids

    # Copy the opcodes except for PUTS without a corresponding GET
    out = io.BytesIO()
    # Write the PROTO header before any framing
    out.write(protoheader)
    pickler = pickle._Pickler(out, proto)
    if proto >= 4:
        pickler.framer.start_framing()
    idx = 0
    for op, arg in opcodes:
        frameless = False
        if op is put:
            if arg not in newids:
                continue
            data = pickler.put(idx)
            newids[arg] = idx
            idx += 1
        elif op is get:
            assert newids[arg] is not None
            data = pickler.get(newids[arg])
        else:
            data = pkl[op:arg]
            frameless = len(data) > pickler.framer._FRAME_SIZE_TARGET
        pickler.framer.commit_frame(force=frameless)
        if frameless:
            pickler.framer.file_write(data)
        else:
            pickler.write(data)
    pickler.framer.end_framing()
    return out.getvalue()
