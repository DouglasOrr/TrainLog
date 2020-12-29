import doctest
import os

import trainlog


def test_doc(tmp_path):
    # Since the doctests create files, use a temporary directory
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        results = doctest.testmod(trainlog)
        assert not results.failed
    finally:
        os.chdir(original_cwd)
