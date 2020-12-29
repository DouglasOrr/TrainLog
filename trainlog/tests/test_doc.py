import doctest
import os
import pathlib
import re
import subprocess
import sys

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


def test_readme(tmp_path):
    readme = pathlib.Path("README.md").read_text()

    pattern = re.compile(r"```python3(.+?)```", re.MULTILINE | re.DOTALL)
    blocks = [r.group(1) for r in pattern.finditer(readme)]
    assert blocks, "readme should contain some testable examples"

    for n, block in enumerate(blocks):
        filename = f"example_{n}.py"
        (tmp_path / filename).write_text(block)
        subprocess.check_call(
            [sys.executable, filename], cwd=tmp_path, env=dict(PYTHONPATH=os.getcwd())
        )
