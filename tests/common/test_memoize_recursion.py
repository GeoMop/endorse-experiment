from endorse.common.memoize2 import ResultCacheFile
import os
import sys
import time
import subprocess


script_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(script_dir, "cache_data")
gen_source_dir = os.path.join(script_dir, "gen_source")


def gen(a, b, c, b_memoize, a2_use, a2):
    os.makedirs(gen_source_dir, exist_ok=True)

    with open(os.path.join(gen_source_dir, "m.py"), "w") as f:
        f.write(template_m.format())

    with open(os.path.join(gen_source_dir, "a.py"), "w") as f:
        f.write(template_a.format(cache_dir, a, "" if a2_use else "#", a2))

    with open(os.path.join(gen_source_dir, "b.py"), "w") as f:
        f.write(template_b.format("" if b_memoize else "#", cache_dir, b))

    with open(os.path.join(gen_source_dir, "c.py"), "w") as f:
        f.write(template_c.format(cache_dir, c))


template_m = '''import a
a.a(1)
'''

template_a = '''from endorse.common.memoize2 import ResultCacheFile, memoize
import b
@memoize(ResultCacheFile("{}"))
def a(x):
    return b.b(1) + {} {} + a2(1)
def a2(x):
    return x + {}
'''

template_b = '''from endorse.common.memoize2 import ResultCacheFile, memoize
import c
{}@memoize(ResultCacheFile("{}"))
def b(x):
    return c.c(1) + {}
'''

template_c = '''from endorse.common.memoize2 import ResultCacheFile, memoize
@memoize(ResultCacheFile("{}"))
def c(x):
    return x + {}
'''


def run_script():
    subprocess.run([sys.executable, "m.py"], cwd=gen_source_dir)


def read_hash(fun_name):
    dir = os.path.join(cache_dir, fun_name)
    if not os.path.isdir(dir):
        return None
    return os.listdir(dir)[0]


def read_hashes():
    return {
        "a": read_hash("a"),
        "b": read_hash("b"),
        "c": read_hash("c")
    }


def run_hash(a, b, c, b_memoize, a2_use, a2):
    ResultCacheFile(cache_dir).clear()
    gen(a, b, c, b_memoize, a2_use, a2)
    run_script()
    hashes = read_hashes()
    time.sleep(1)
    return hashes


def test_recursion():
    # same code
    h1 = run_hash(1, 1, 1, True, False, 1)
    h2 = run_hash(1, 1, 1, True, False, 1)
    assert h1 == h2

    # difference in first function level
    h3 = run_hash(2, 1, 1, True, False, 1)
    assert h1["a"] != h3["a"]
    assert h1["b"] == h3["b"]
    assert h1["c"] == h3["c"]

    # difference in last function level
    h4 = run_hash(1, 1, 2, True, False, 1)
    assert h1["a"] != h4["a"]
    assert h1["b"] != h4["b"]
    assert h1["c"] != h4["c"]

    # difference in last function level, middle function without memoize decorator
    h5 = run_hash(1, 1, 1, False, False, 1)
    h6 = run_hash(1, 1, 2, False, False, 1)
    assert h5["a"] == h6["a"]
    assert h5["c"] != h6["c"]

    # difference in not decorated function in same module with decorated function
    h7 = run_hash(1, 1, 1, False, True, 1)
    h8 = run_hash(1, 1, 2, False, True, 2)
    assert h7["a"] != h8["a"]


#test_recursion()
