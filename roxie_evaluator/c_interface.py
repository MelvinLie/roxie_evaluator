import ctypes
import os

def test_yourself():
    print(os.getcwd())
    testlib = ctypes.CDLL(os.getcwd() + '/roxie_evaluator/c-algorithms/test_lib.dll')
    testlib.myprint()