import os
import sys
import time
import subprocess
from joblib import Memory


script_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(script_dir, "cache_data")
gen_source_dir = os.path.join(script_dir, "gen_source")


memory = Memory(cache_dir, verbose=0)


def gen(a, b, c, b_memoize, a2_use, a2):
    os.makedirs(gen_source_dir, exist_ok=True)

    with open(os.path.join(gen_source_dir, "m.py"), "w") as f:
        f.write(template_m.format())

    with open(os.path.join(gen_source_dir, "a.py"), "w") as f:
        f.write(template_a.format(cache_dir, a, "" if a2_use else "#", a2))

    with open(os.path.join(gen_source_dir, "b.py"), "w") as f:
        f.write(template_b.format(cache_dir, "" if b_memoize else "#", b))

    with open(os.path.join(gen_source_dir, "c.py"), "w") as f:
        f.write(template_c.format(cache_dir, c))


template_m = '''import a
a.a(1)
'''

template_a = '''import b
import time
from joblib import Memory
memory = Memory("{}", verbose=0)
@memory.cache
def a(x):
    time.sleep(1)
    return b.b(1) + {} {} + a2(1)
def a2(x):
    return x + {}
'''

template_b = '''import c
from joblib import Memory
memory = Memory("{}", verbose=0)
{}@memory.cache
def b(x):
    return c.c(1) + {}
'''

template_c = '''from joblib import Memory
memory = Memory("{}", verbose=0)
@memory.cache
def c(x):
    return x + {}
'''


def run_script():
    subprocess.run([sys.executable, "m.py"], cwd=gen_source_dir)


def run(a, b, c, b_memoize, a2_use, a2):
    gen(a, b, c, b_memoize, a2_use, a2)
    t = time.time()
    run_script()
    return time.time() - t > 0.5


def test_recursion():
    # same code
    memory.clear()
    assert run(1, 1, 1, True, False, 1)
    assert not run(1, 1, 1, True, False, 1)

    # difference in first function level
    memory.clear()
    assert run(1, 1, 1, True, False, 1)
    assert run(2, 1, 1, True, False, 1)

    # difference in last function level
    memory.clear()
    assert run(1, 1, 1, True, False, 1)
    #assert run(1, 1, 2, True, False, 1)

    # difference in last function level, middle function without memoize decorator
    memory.clear()
    assert run(1, 1, 1, False, False, 1)
    assert not run(1, 1, 2, False, False, 1)

    # difference in not decorated function in same module with decorated function
    memory.clear()
    assert run(1, 1, 1, True, True, 1)
    #assert run(1, 1, 1, True, True, 2)
