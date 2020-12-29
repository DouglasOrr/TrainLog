"""
# Example (PyTorch)

Adding logging to a PyTorch training loop is straightforward. We `trainlog.logger.open`
the log in a `with` scope, and use `trainlog.logger.Log.add` to emit events.

```python
# training.py

import torch as T
import trainlog as L  #<==

def generate_data(n):
    "Create a simple dimension-permutation dataset."
    dim = 16
    true_weight = T.eye(dim).roll(1)
    xs = T.randn(n, dim)
    ys = xs @ true_weight + 0.1 * T.randn(n, dim)
    return xs, ys

train_x, train_y = generate_data(100)
valid_x, valid_y = generate_data(100)

def run_training(learning_rate, logger):
    batch_size = 10
    module = T.nn.Linear(train_x.shape[1], train_y.shape[1])
    opt = T.optim.SGD(module.parameters(), learning_rate * batch_size)
    for _ in range(10):
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]
            opt.zero_grad()
            loss = T.nn.functional.mse_loss(batch_y, module(batch_x))
            loss.backward()
            opt.step()
            logger.add("step", loss=float(loss))  #<==
        valid_loss = T.nn.functional.mse_loss(valid_y, module(valid_x))
        logger.add("valid", loss=float(valid_loss))  #<==

for n, lr in enumerate([0.1, 0.01, 0.001]):
    with L.logger.open(f"log_{n}.jsonl", learning_rate=lr) as logger:  #<==
        run_training(lr, logger)
```

Now we're ready to analyse the logs. We use `trainlog.logs.glob` to find log files.
Note the `.gzip` extension, by default the logs are gzipped when training finishes.
Then, `trainlog.logs.LogSet.apply` copies the learning rate from the header into each
event, and counts the number of previous `step` events. This makes it easier to use
pandas to process & plot them.

Finally, `log["valid"]` selects just the validation events from the logs and
`trainlog.logs.LogSet.to_pandas` gives us a
[pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
to use for further analysis.

```python
# analysis.py

import trainlog as L
import trainlog.ops as O
import matplotlib.pyplot as plt

log = L.logs.glob("log*.jsonl.gz")
log = log.apply(O.header("learning_rate"), O.count("step"))
df = log["valid"].to_pandas()

for lr, g in df.groupby("learning_rate"):
    g.plot(x="step", y="loss", ax=plt.gca(), label=str(lr))
plt.savefig("analysis.png")
```
"""
