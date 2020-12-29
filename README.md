# Trainlog

A simple logging library, designed for deep learning.

Trainlog's aim is to make it easy to create a single file containing all
key info about a training run. We prioritize convenience and ease of use
over efficiency.

## Example

Code to generate the log looks like this:

```python
import trainlog as L

with L.logger.open("log.jsonl", lr=0.01) as logger:
    logger.add("step", loss=10)
    logger.add("step", loss=5)
    logger.add("step", loss=2.5)
```

You can inspect the log using standard tools `gzip -cd log.jsonl.gz | jq -Cc . | less -R`.

Code to analyse the log looks like this:

```python
import trainlog as L
import trainlog.ops as O

log = L.logs.open("log.jsonl.gz")
log = log.apply(O.header("lr"), O.count("step"))
df = log["step"].to_pandas()
print(df)
```

## See also

 - [Design rationale](doc/design.md)
 - [Developing trainlog yourself](doc/development.md)
