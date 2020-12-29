import doctest
import os
import pathlib
import re
import subprocess
import sys

import pytest

import trainlog
import trainlog.examples.pytorch


def test_doc(tmp_path):
    # Since the doctests create files, use a temporary directory
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        results = doctest.testmod(trainlog)
        assert not results.failed
    finally:
        os.chdir(original_cwd)


def get_python_blocks(doc):
    pattern = re.compile(r"```python(.+?)```", re.MULTILINE | re.DOTALL)
    blocks = [r.group(1) for r in pattern.finditer(doc)]
    assert blocks, "expected at least one testable code block"
    return blocks


def test_readme(tmp_path):
    readme = pathlib.Path("README.md").read_text()
    for n, block in enumerate(get_python_blocks(readme)):
        filename = f"readme_{n}.py"
        (tmp_path / filename).write_text(block)
        print()
        subprocess.check_call(
            [sys.executable, filename], cwd=tmp_path, env=dict(PYTHONPATH=os.getcwd())
        )


@pytest.mark.example
def test_example(tmp_path):  # pragma: no cover
    doc = trainlog.examples.pytorch.__doc__
    blocks = get_python_blocks(doc)

    # Each block is a separate python script, but we create a single bash script
    # to install dependencies & run everything
    script = ["pip install pandas matplotlib"]
    for n, block in enumerate(blocks):
        filename = f"example_{n}.py"
        (tmp_path / filename).write_text(block)
        script.append(f"python {filename}")
    (tmp_path / "test.sh").write_text("\n".join(script))

    subprocess.check_call(
        [
            "docker",
            "run",
            "--rm",
            "--interactive",
            f"--volume={tmp_path.absolute()}:/work",
            f"--volume={os.getcwd()}:/app",
            "--env=PYTHONPATH=/app",
            "--workdir=/work",
            "pytorch/pytorch",
            "bash",
            "test.sh",
        ]
    )
