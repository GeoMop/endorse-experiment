from endorse.common.memoize2 import pickle_remove_duplicit
import pickle
import pickletools


def test_pickle_remove_duplicit():
    a = "he"
    b = "llo"
    c = a + b
    d = a + b

    bin1 = pickle.dumps((c, c))
    bin2 = pickle.dumps((c, d))
    bin3 = pickle.dumps((c, c, c, c))
    bin4 = pickle.dumps((c, d, c, d))

    assert pickletools.optimize(bin1) == pickle_remove_duplicit(bin2)
    assert pickletools.optimize(bin3) == pickle_remove_duplicit(bin4)
