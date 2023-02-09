# NCE CD+CIS

This repo contains the official implementation of the methods described in the paper "On the connection between Noise-Contrastive Estimation and Contrastive Divergence".

## Setup

This repo uses `Pipenv` to manage dependencies.
To install, run

```
pipenv install
```

or install the dependencies listed in `Pipfile` in another fashion.

## Reproduce results

### Adaptive proposal distribution toy example

To generate the plot in Figure 1, run:

```
python src/experiments/adaptive_proposal.py --num-runs=20 --num_epochs=2000
```

### Ring model experiments

To generate the plot in Figure 2, run:

```
python src/experiments/TODO
```

To generate the plot in Figure 3, run:

```
python src/experiments/TODO
```

## TODO

- Add input x to `BaseModel`
- Rename `noise_distr` -> `proposal_distr`
- NoiseDistr.sample should take an int as sample size, not a torch size (confusing).
- Weird reshapes to use conditional model. Move this to interior in cond distr. (interleave-function removed in cond. MVN model (do broadcasting instead)
- idx param in `part_fn` interface
