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
            m.update(pickle_remove_duplicit(cloudpickle.dumps((func, args, kwargs))))
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


def pickle_remove_duplicit(p):
    'Optimize a pickle string by removing unused PUT opcodes and duplicated unicode strings'
    put = 'PUT'
    get = 'GET'
    oldids = set()          # set of all PUT ids
    newids = {}             # set of ids used by a GET opcode
    opcodes = []            # (op, idx) or (pos, end_pos)
    proto = 0
    protoheader = b''
    strings = {}
    memo_map = {}
    last_opcode_name = ""
    last_arg = None
    for opcode, arg, pos, end_pos in pickletools._genops(p, yield_end_pos=True):
        if 'PUT' in opcode.name:
            oldids.add(arg)
            opcodes.append((put, arg))
        elif opcode.name == 'MEMOIZE':
            idx = len(oldids)
            if 'BINUNICODE' in last_opcode_name:
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
            pass
        elif 'GET' in opcode.name:
            if opcode.proto > proto:
                proto = opcode.proto
            if arg in memo_map:
                arg = memo_map[arg]
            newids[arg] = None
            opcodes.append((get, arg))
        elif opcode.name == 'PROTO':
            if arg > proto:
                proto = arg
            if pos == 0:
                protoheader = p[pos:end_pos]
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
            data = pickler.get(newids[arg])
        else:
            data = p[op:arg]
            frameless = len(data) > pickler.framer._FRAME_SIZE_TARGET
        pickler.framer.commit_frame(force=frameless)
        if frameless:
            pickler.framer.file_write(data)
        else:
            pickler.write(data)
    pickler.framer.end_framing()
    return out.getvalue()
