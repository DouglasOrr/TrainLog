"""A simple logging library, designed for deep learning.

Trainlog's aim is to make it easy to create a single file containing all
key info about a training run. We prioritize convenience and ease of use
over efficiency.

**Example**

This is a minimal example of training and analysis using trainlog.
For a more complete example, see `trainlog.examples.pytorch`.

First some aliased imports. The second is optional (`L.ops.*` would also
work).

>>> import trainlog as L
>>> import trainlog.ops as O

In training, we open a log, add some custom info to the header event,
`name="example"`, and emit some training step events of kind "step".

>>> with L.logger.open("log.jsonl", name="example") as logger:
...     logger.add("step", loss=10)
...     logger.add("step", loss=5)
...     logger.add("step", loss=2.5)

In post-hoc analysis we open the log file, which has been gzipped at the
end of training, hence the `.gz` extension.

>>> log = L.logs.open("log.jsonl.gz")

We transform the log by copying `"name"` from the header, and computing a
step counter. Then we select just the `"step"` events and convert them to a
pandas DataFrame for further analysis.

>>> log = log.apply(O.header("name"), O.count("step"))
>>>
>>> df = log["step"].to_pandas()
>>> df[["step", "loss"]]
   step  loss
0     0  10.0
1     1   5.0
2     2   2.5
"""

from . import io, logger, logs, ops  # noqa: F401
from ._version import __version__  # noqa: F401

__pdoc__ = dict(tests=False)
