# NCE CD+CIS

This is the implementation of the methods described in the paper "PAPER TITLE".

## Setup

This repo uses `Pipenv` to manage dependencies.
To install, run

```
pipenv install
```

## Reproduce results

### Adaptive proposal distribution toy example

To generate the plot in Figure 1, run:

```
python src/experiments/adaptive_proposal.py --num-runs=20 --num_epochs=2000
```

## TODO

- Add input x to `BaseModel`
- Rename `noise_distr` -> `proposal_distr`
- Weird reshapes to use conditional model. Move this to interior in cond distr. (interleave-funktion borttagen i cond. MVN model (kör broadcasting istället)
- ~Remove idx param in `inner_crit`~
- idx param in `part_fn` interface
- ~Persistent CNCE inherits from CNCE~
