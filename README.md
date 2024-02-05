# NCE CD+CIS

This is the implementation of the methods described in the paper "On the Connection Between Noise Contrastive Estimation and Contrastive Divergence".

## Setup

This repo uses `Pipenv` to manage dependencies.
To install, run

```
pipenv install
```

## Reproduce results

### Adaptive proposal distribution toy example

To generate the plot in Figure (Left), run:

```
python experiments/adaptive_proposal.py
```

### Ring model experiments (use of MH acceptance probability in CNCE) 

To generate the plot in Figure 1 (Middle-Right), run:

```
python experiments/acceptance_probability.py
```

### Autoregressive EBM experiments 

To reproduce results in Table 1. 

Power:

```
python experiments/aem --criterion CRIT --n_total_steps 1000000 
```

Gas:
 -m nbs.aem_experiment --criterion 'is' --dataset_name 'miniboone'  --dropout_probability_energy_net 0.5 --dropout_probability_made 0.5 --train_batch_size 128 --val_batch_size 128 --n_total_steps 50000 --alpha_warm_up_steps 0 --val_frac 1.0 | tee miniboone_exp_is.txt

```
python experiments/aem --criterion CRIT --dataset_name 'gas' --dropout_probability_energy_net 0.0 --dropout_probability_made 0.0 --activation_energy_net 'tanh' --n_total_steps 600000 
```

Hepmass:

```
python experiments/aem --criterion CRIT --dataset_name 'hepmass' --n_total_steps 200000
```

Miniboone: 

```
python experiments/aem --criterion CRIT --dataset_name 'miniboone'  --dropout_probability_energy_net 0.5 --dropout_probability_made 0.5 --train_batch_size 128 --val_batch_size 128 --n_total_steps 300000 --val_frac 1.0
```

BSDS300:

```
python experiments/aem --criterion CRIT --dataset_name 'bsds300' --hidden_dim_made 512 --train_batch_size 128  --val_batch_size 128  --n_total_steps 600000 
```

Replace CRIT with designated criterion ('is'/'cis'/csmc').


To evaluate the log. likelihood for all critera^1:

Power:

```
python experiments/aem_eval_log_likelihood --val_frac 1.0 --n_importance_samples 5000000
```

Gas:

```
python experiments/aem_eval_log_likelihood --dataset_name 'gas' --dropout_probability_energy_net 0.0 --dropout_probability_made 0.0 --activation_energy_net 'tanh' --val_frac 1.0 --n_importance_samples 5000000
```

Hepmass:

```
python experiments/aem_eval_log_likelihood --dataset_name 'hepmass' --val_frac 1.0 --n_importance_samples 5000000
```

Miniboone:

```
python experiments/aem_eval_log_likelihood --dataset_name 'miniboone'  --dropout_probability_energy_net 0.5 --dropout_probability_made 0.5 --train_batch_size 128 --val_batch_size 128 --val_frac 1.0 --n_importance_samples 5000000
```

BSDS300:

```
python experiments/aem_eval_log_likelihood --dataset_name 'bsds300' --hidden_dim_made 512 --train_batch_size 128  --val_batch_size 128 --val_frac 1.0 --n_importance_samples 5000000
```



To evaluate the Wasserstein distance for all critera:


Power:

```
python experiments/aem_eval_wasserstein --val_frac 1.0 --n_importance_samples 10000
```

Gas:

```
python experiments/aem_eval_wasserstein --dataset_name 'gas' --dropout_probability_energy_net 0.0 --dropout_probability_made 0.0 --activation_energy_net 'tanh' --val_frac 1.0 --n_importance_samples 10000
```

Hepmass:

```
python experiments/aem_eval_wasserstein --dataset_name 'hepmass' --val_frac 1.0 --n_importance_samples 10000
```

Miniboone:

```
python experiments/aem_eval_wasserstein --dataset_name 'miniboone'  --dropout_probability_energy_net 0.5 --dropout_probability_made 0.5 --train_batch_size 128 --val_batch_size 128 --val_frac 1.0 --n_importance_samples 2000
```

BSDS300:

```
python experiments/aem_eval_log_likelihood --dataset_name 'bsds300' --hidden_dim_made 512 --train_batch_size 128  --val_batch_size 128 --val_frac 1.0 --n_importance_samples 10000
```



```
python experiments/aem_eval_wasserstein.py
```



^1 Evaluation expects that all models for the given dataset have been trained (IS, CIS, CSMC for Power, Gas, Hepmass and CIS, CSMC for Miniboone, BSDS300). 


## TODO

- Add input x to `BaseModel`
- Rename `noise_distr` -> `proposal_distr`
- Weird reshapes to use conditional model. Move this to interior in cond distr. (interleave-funktion borttagen i cond. MVN model (kör broadcasting istället)
- ~Remove idx param in `inner_crit`~
- idx param in `part_fn` interface
- ~Persistent CNCE inherits from CNCE~ (DONE?)


