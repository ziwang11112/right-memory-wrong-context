# Submission Zero-Call Diagnostics

These post-hoc diagnostics reuse the frozen top-20 natural rankings, exact
dense scores, released required anchors, and primary 0.8 recall target. They
make no provider, reader, judge, or paid calls and perform no retuning.

## Evaluator-label missingness

Established composite admissibility judgments are hidden deterministically
after retrieval. Routes, anchors, recall, feasibility, and underlying labels
remain fixed. The exercise isolates the empirical role of three-valued
accounting; it is not a model of deployment-time metadata corruption.

| Arm | Hidden fraction | Coverage | Lower risk | Upper risk | Bound width |
| --- | ---: | ---: | ---: | ---: | ---: |
| global dense | 0.00 | 0.992 | 0.187 | 0.195 | 0.008 |
| namespace dense | 0.00 | 0.992 | 0.118 | 0.126 | 0.008 |
| global dense | 0.20 | 0.786 | 0.150 | 0.364 | 0.214 |
| namespace dense | 0.20 | 0.793 | 0.095 | 0.302 | 0.207 |
| global dense | 0.50 | 0.496 | 0.093 | 0.597 | 0.504 |
| namespace dense | 0.50 | 0.495 | 0.060 | 0.565 | 0.505 |
| global dense | 0.90 | 0.095 | 0.019 | 0.924 | 0.905 |
| namespace dense | 0.90 | 0.101 | 0.012 | 0.911 | 0.899 |

## Gold-preserving same-size support

The oracle control retains every released required anchor, then samples
non-anchors without replacement until its candidate count exactly matches
the query's namespace support. It separates anchor retention plus support
size from the identity-aligned composition supplied by a namespace. Because
it reads released gold, it is diagnostic and cannot be deployed.

| Arm | Uses gold | Recall | Feasible | Penalized upper risk | Candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| Global dense | false | 0.432 | 0.237 | 0.798 | 90122 |
| Random same-size | false | 0.061 | 0.024 | 0.978 | 1540 |
| Gold-preserving same-size | true | 0.840 | 0.747 | 0.390 | 1551 |
| Namespace pre-filter | false | 0.533 | 0.311 | 0.713 | 1551 |

All outputs are content-free aggregates. No query, memory, or namespace
identifier; text; embedding; prompt; or response is written.
