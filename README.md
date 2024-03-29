# NCE CD+CIS

This is the implementation of the methods described in the paper "On the Connection Between Noise Contrastive Estimation and Contrastive Divergence",
accepted for publication in AISTATS 2024.

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

#### Training

```
# Power:
python experiments/aem.py --criterion CRIT --n_total_steps 1000000 
# Gas:
python experiments/aem.py --criterion CRIT --dataset_name 'gas' --dropout_probability_energy_net 0.0 --dropout_probability_made 0.0 --activation_energy_net 'tanh' --n_total_steps 600000 
# Hepmass:
python experiments/aem.py --criterion CRIT --dataset_name 'hepmass' --n_total_steps 200000
# Miniboone: 
python experiments/aem.py --criterion CRIT --dataset_name 'miniboone'  --dropout_probability_energy_net 0.5 --dropout_probability_made 0.5 --train_batch_size 128 --val_batch_size 128 --n_total_steps 300000 --val_frac 1.0
# BSDS300:
python experiments/aem.py --criterion CRIT --dataset_name 'bsds300' --hidden_dim_made 512 --train_batch_size 128  --val_batch_size 128  --n_total_steps 600000 
```

Replace CRIT with designated criterion ('is'/'cis'/csmc').

#### Evaluation

To evaluate the log. likelihood for all criteria[^1]:


```
# Power:
python experiments/aem_eval_log_likelihood.py --val_frac 1.0 --n_importance_samples 5000000
# Gas:
python experiments/aem_eval_log_likelihood.py --dataset_name 'gas' --dropout_probability_energy_net 0.0 --dropout_probability_made 0.0 --activation_energy_net 'tanh' --val_frac 1.0 --n_importance_samples 5000000
# Hepmass:
python experiments/aem_eval_log_likelihood.py --dataset_name 'hepmass' --val_frac 1.0 --n_importance_samples 5000000
# Miniboone:
python experiments/aem_eval_log_likelihood.py --dataset_name 'miniboone'  --dropout_probability_energy_net 0.5 --dropout_probability_made 0.5 --train_batch_size 128 --val_batch_size 128 --val_frac 1.0 --n_importance_samples 5000000
# BSDS300:
python experiments/aem_eval_log_likelihood.py --dataset_name 'bsds300' --hidden_dim_made 512 --train_batch_size 128  --val_batch_size 128 --val_frac 1.0 --n_importance_samples 5000000
```


To evaluate the Wasserstein distance for all criteria:



```
# Power:
python experiments/aem_eval_wasserstein.py --val_frac 1.0 --n_importance_samples 10000
# Gas:
python experiments/aem_eval_wasserstein.py --dataset_name 'gas' --dropout_probability_energy_net 0.0 --dropout_probability_made 0.0 --activation_energy_net 'tanh' --val_frac 1.0 --n_importance_samples 10000
# Hepmass:
python experiments/aem_eval_wasserstein.py --dataset_name 'hepmass' --val_frac 1.0 --n_importance_samples 10000
# Miniboone:
python experiments/aem_eval_wasserstein.py --dataset_name 'miniboone'  --dropout_probability_energy_net 0.5 --dropout_probability_made 0.5 --train_batch_size 128 --val_batch_size 128 --val_frac 1.0 --n_importance_samples 2000
# BSDS300:
python experiments/aem_eval_wasserstein.py --dataset_name 'bsds300' --hidden_dim_made 512 --train_batch_size 128  --val_batch_size 128 --val_frac 1.0 --n_importance_samples 10000
```


[^1]: Evaluation expects that all models for the given dataset have been trained (IS, CIS, CSMC for Power, Gas, Hepmass and CIS, CSMC for Miniboone, BSDS300). 
