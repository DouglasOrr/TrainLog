# TrainLog design decisions

This document describes some of the design decisions behind trainlog. You may not agree, but there was at least a reason...

## JSONlines + GZip

We want the file format to be portable and simple to read/write. JSON, for all its rough edges, is popular and easy to read/write. At the same time, it seems slightly silly to save the string `"loss":` thousands of times; gzip compresses this boilerplate away.

## Write plain, gzip on close()

We'd ideally like to write directly into a gzip file, but this can make the file non-readable while it is being written, at least if using Python's `gzip` module. This is a shame when we want to inspect in-progress training runs, so we choose to write in plain `.jsonl`, then convert to `.jsonl.gz` when the log is closed.

## No explicit schema

We choose not to define an explicit schema for log events. This is for sake of ease of use & to match the expectations of the Python community. No explicit schema means no schema definition language to learn, and it's very simple to add information into the log (for example quickly adding data to assist one-off debugging).

## Single file

We prefer to write all relevant information to a single file. E.g. instead of `(train.jsonl, valid.jsonl, debug.jsonl)` we have `log.jsonl` with heterogenous events. This makes it easy to be sure the information came from the same run, and easier to manage the files. We believe any processing inefficiency (e.g. you're only interested in the per-epoch validation results but have to skip past all per-batch train events) can be handled by creating vertically partitioned caches.

## Header line

Experiment runs usually associate substantial "one off" information - hyperparameters, execution environment, etc. This information can be stored separately (e.g. `(settings.json, log.jsonl)`), but in keeping with our _single file_ principle we'd like to keep it together. A header line at the beginning of the log file provides a standard place for this.

## A custom `trainlog.ops` library for transforming logs

Event logs containing different types of events can be slightly annoying to analyse. E.g. a sensible logging scheme would be to put experiment hyperparameters in the _header_ event, training steps in _step_ events and end-of-epoch validation in _valid_ events.

When we want to show end-of-epoch training & validation curves using pandas, the easiest way is to have a single `DataFrame` containing all experiments, with each row containing the relevant hyperparameters, step count so far, and training performance. This is a denormalized form of the logs. Although it's not too hard to denormalize the logs when being read using standard Python, generators & loops, we prefer to make it even easier, in order to remove any temptation to store the logs in denormalized form. So `trainlog.ops` aims to make common denormalizations easy, including step counting & copying from header events.
