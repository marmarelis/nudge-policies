import pandas as pd
import numpy as np
import ot
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from scipy.sparse import csc_array
from scipy.stats import ttest_1samp, wilcoxon, entropy
from tqdm import tqdm, trange
import pickle as pkl
import os

if os.path.isfile("../style.mplstyle"):
  from matplotlib import pyplot as plt
  plt.ion()
  plt.style.use("../style.mplstyle")


import torch as tr
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt

# for all nuisances AND learned policies
class NeuralModel(nn.Module):
  def __init__(self, n_inputs, n_outputs, width=50, depth=2):
    super().__init__()
    self.first = nn.Linear(n_inputs, width if depth>0 else 2)
    self.middle = nn.ModuleList([
      nn.Linear(width, width) for _ in range(depth-1) ])
    self.last = nn.Linear(width, n_outputs) if depth>0 else None
    self.dropout = nn.Dropout(p=0.05)

  def forward(self, *inputs):
    input = tr.cat(inputs, dim=-1)
    if self.last is None:
      return self.first(input)
    inner = self.dropout( F.silu(self.first(input)) )
    for layer in self.middle:
      inner = self.dropout( F.silu(layer(inner)) )
    return self.last(inner).squeeze() # squeeze for single output
  
# drive home that these are pretty standard becasue we are NOT studying better predictor model architectures

batch_size = 256
n_epochs = 1024
outcome_learning_rate = 1e-3
propensity_learning_rate = 1e-4
policy_learning_rate = 1e-3
naive_policy_learning_rate = 1e-3
weight_decay = 0
n_splits = 5
n_policy_splits = 5
nudge_sample_size = 1024

## THIS DATASET IS TAKEN FROM https://github.com/ioanabica/SCIGAN
with open("tcga.p", "rb") as f:
  tcga = pkl.load(f)['rnaseq']
n_units, n_genes = tcga.shape

n_covariates = 60
n_treatments = 4
n_variables = n_treatments + n_covariates
default_nudge_radius = 0.5

def generate_variables(data, device="cpu"):
  data = tr.from_numpy(data).float().to(device)
  treatment_proj = tr.randn(n_genes, n_treatments, device=device)
  covariate_proj = tr.randn(n_genes, n_covariates, device=device)
  entangled_proj = tr.randn(n_covariates, n_treatments, device=device)
  treatment_proj += covariate_proj @ entangled_proj #/ np.sqrt(n_covariates) # entangle the two. sqrt would keep emphasis equal
  treatments = data @ treatment_proj
  covariates = data @ covariate_proj
  treatments = (treatments - treatments.mean(dim=0)) / treatments.std(dim=0)
  covariates = (covariates - covariates.mean(dim=0)) / covariates.std(dim=0)
  return treatments, covariates

# for ground truth, take linear combinations of variables and feed into sigmoid/cosine (nice and simple)
def generate_mixing_link(n_nonlinearities=4, device="cpu"):
  mixing = tr.randn(n_variables, n_nonlinearities, device=device) / np.sqrt(n_variables)
  # upweigh treatments while keeping stdev of projection near 1
  mixing[:n_treatments] *= np.sqrt(n_covariates/n_treatments)
  mixing[n_treatments:] /= np.sqrt(n_covariates/n_treatments)
  return mixing

# set noise to zero for test-set evaluations
def generate_outcomes(mixing, treatments, covariates, scaling=1, noise=0.10, device="cpu"):
  variables = tr.cat([treatments, covariates], dim=1)
  projection = variables @ mixing
  n_bases = mixing.shape[1]
  #output = tr.mean( tr.cos(scaling*np.pi * projection), dim=1)
  response = tr.mean( (scaling * projection) ** tr.arange(1, n_bases+1), dim=1)
  gate = tr.mean( (scaling * variables) ** 2 ) ** ((n_bases+2)/2) # last one is L2 norm
  output = -(response+gate)
  # I added the exponential/softplus so that runaway nudges go to 0, not -inf
  return F.softplus( output  +  noise * tr.randn(output.shape[0], device=device) )


def learn_policy(effects, reg=0, matched=True): # nudges x units
  assert reg >= 0
  original_dtype = effects.dtype
  n_nudges, n_units = effects.shape
  length = n_nudges * n_units
  # units inner, nudges outer
  coefficients = effects.reshape(-1).astype(np.float32) - reg
  unit_sums = np.zeros((n_units, length), dtype=np.float32)
  nudge_sums = np.zeros((n_nudges, length), dtype=np.float32)
  for unit in range(n_units):
    indices = range(unit, length, n_units)
    unit_sums[unit, indices] = 1/n_nudges
  for nudge in range(n_nudges):
    indices = range(nudge*n_units, (nudge+1)*n_units)
    nudge_sums[nudge, indices] = 1/n_units if matched else 0
  constraints = csc_array( np.concatenate((unit_sums, nudge_sums), axis=0) )
  result = linprog(-coefficients, # maximize
    A_ub=constraints, b_ub=np.ones(n_units+n_nudges, dtype=np.float32),
    bounds=(0, None), method="highs-ipm") # more scalable?
  policy = result.x.reshape(n_nudges, n_units).astype(original_dtype)
  policy_effect = np.mean(policy * effects)
  return policy, policy_effect


# I like neural networks to be ephemeral.
def experiment(seed, nudge_radius=default_nudge_radius, device="cpu"):
  np.random.seed(seed)
  tr.manual_seed(seed)
  ### setup
  insample = np.random.choice(range(n_units), size=int(0.8*n_units), replace=False)
  outsample = np.setdiff1d(range(n_units), insample)
  split_indices = np.linspace(0, len(insample), n_splits+1).astype(int)
  partitions = tr.tensor(np.digitize(insample, split_indices[1:-1]), device=device)
  treatments, covariates = generate_variables(tcga, device=device)
  mixing = generate_mixing_link(device=device)
  outcomes = generate_outcomes(mixing, treatments, covariates, device=device)
  ### estimate nuisances
  outcome_models = []
  propensity_models = []
  temperatures = []
  for split in range(n_splits):
    print(f"=== Estimation Split #{split+1} ===")
    insample_test = insample[ split_indices[split] : split_indices[split+1] ]
    insample_train = np.setdiff1d(insample, insample_test)
    outcome_model = NeuralModel(n_inputs=n_variables, n_outputs=1).to(device)
    criterion = nn.MSELoss()
    optimizer = opt.Adam(outcome_model.parameters(),
      lr=outcome_learning_rate, weight_decay=weight_decay)
    valid_treatments = treatments[insample_test, :]
    valid_covariates = covariates[insample_test, :]
    valid_outcomes = outcomes[insample_test]
    with trange(n_epochs, desc="Outcome Model") as iterator:
      for epoch in iterator:
        epoch_train_loss = []
        for batch_index in range(0, len(insample_train), batch_size):
          optimizer.zero_grad() # I forgot these earlier...
          batch_treatments = treatments[insample_train[batch_index : batch_index+batch_size], :]
          batch_covariates = covariates[insample_train[batch_index : batch_index+batch_size], :]
          batch_outcomes = outcomes[insample_train[batch_index : batch_index+batch_size]]
          output = outcome_model(batch_treatments, batch_covariates)
          train_loss = criterion(output, batch_outcomes)
          epoch_train_loss.append(train_loss.item())
          train_loss.backward()
          optimizer.step()
        valid_output = outcome_model(valid_treatments, valid_covariates)
        epoch_valid_loss = criterion(valid_output, valid_outcomes).item()
        iterator.set_postfix(train_loss=np.mean(epoch_train_loss), valid_loss=np.mean(epoch_valid_loss))
    propensity_model = NeuralModel(n_inputs=n_variables, n_outputs=n_treatments).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = opt.Adam(propensity_model.parameters(),
      lr=propensity_learning_rate, weight_decay=weight_decay)
    with trange(n_epochs, desc="Propensity Model") as iterator:
      for epoch in iterator:
        epoch_train_loss = []
        for batch_index in range(0, len(insample_train), batch_size):
          optimizer.zero_grad()
          batch_treatments = treatments[insample_train[batch_index : batch_index+batch_size], :]
          batch_covariates = covariates[insample_train[batch_index : batch_index+batch_size], :]
          this_batch_size = batch_treatments.shape[0]
          label = tr.randint(2, (this_batch_size, 1), dtype=tr.float, device=device)
          noise = nudge_radius * tr.randn((this_batch_size, n_treatments), device=device)
          noisy_treatments = batch_treatments  +  noise * label
          output = propensity_model(noisy_treatments, batch_covariates)
          logit = tr.sum(output * noise, dim=-1)
          train_loss = criterion(logit, label[:, 0])
          epoch_train_loss.append(train_loss.item())
          train_loss.backward()
          optimizer.step()
        valid_label = tr.randint(2, (len(insample_test), 1), dtype=tr.float, device=device)
        valid_noise = nudge_radius * tr.randn((len(insample_test), n_treatments), device=device)
        valid_noisy_treatments = valid_treatments  +  valid_noise * valid_label
        valid_output = propensity_model(valid_noisy_treatments, valid_covariates)
        valid_logit = tr.sum(valid_output * valid_noise, dim=-1)
        epoch_valid_loss = criterion(valid_logit, valid_label[:, 0]).item()
        iterator.set_postfix(train_loss=np.mean(epoch_train_loss), valid_loss=np.mean(epoch_valid_loss))
    # temperature calibration
    cal_nudges = nudge_radius * tr.randn((len(insample_test), n_treatments), device=device)
    nudged_treatments = valid_treatments + cal_nudges
    original_propensity = propensity_model(valid_treatments, valid_covariates)
    nudged_propensity = propensity_model(nudged_treatments, valid_covariates)
    original_logit = tr.sum(original_propensity * cal_nudges, dim=-1).double()
    nudged_logit = tr.sum(nudged_propensity * cal_nudges, dim=-1).double()
    eval_temp = lambda t: -tr.mean(
      tr.log(tr.sigmoid(nudged_logit/t)) + tr.log(1-tr.sigmoid(original_logit/t)) )
    temperature = 1.0
    temp_delta = 0.01
    best_loss = eval_temp(temperature)
    for _ in range(10_000):
      current_loss = eval_temp(temperature+temp_delta)
      if current_loss > best_loss:
        break
      best_loss = current_loss
      temperature += temp_delta
    print(f"Optimized temperature to {temperature}.")
    outcome_models.append( outcome_model.eval() )
    propensity_models.append( propensity_model.eval() )
    temperatures.append(temperature)
  ### learn policy
  shuffled_indices = np.arange(len(insample))
  np.random.shuffle(shuffled_indices)
  policy_split_indices = np.linspace(0, len(insample), n_policy_splits+1).astype(int)
  regression_covariates = []
  regression_policies = []
  regression_direct_policies = []
  with trange(n_policy_splits, desc="=== Policy Learner") as iterator:
    for policy_split in iterator:
      policy_indices = shuffled_indices[
        policy_split_indices[policy_split] : policy_split_indices[policy_split+1] ]
      nuisance_ids = np.digitize(policy_indices, split_indices[1:-1]) # which split do the above belong to? for referring to nuisances
      nudges = nudge_radius * tr.randn((nudge_sample_size, n_treatments)) # on cpu
      split_pseudo_outcomes = []
      split_effect_predictions = []
      split_covariates = []
      for split in range(n_splits):
        indices = insample[ policy_indices[nuisance_ids == split] ]
        tiling = (1, nudge_sample_size, 1)
        policy_treatments = treatments[indices, :][:, None, :].tile(tiling).cpu()
        policy_covariates = covariates[indices, :][:, None, :].tile(tiling).cpu()
        policy_outcomes = outcomes[indices][:, None].tile(tiling[:-1]).cpu()
        nudged_treatments = policy_treatments + nudges[None, :, :] # units x nudges x n_treatments
        outcome_model = outcome_models[split].cpu() # for now, because inputs might not fit on GPU entirely
        propensity_model = propensity_models[split].cpu()
        temperature = temperatures[split]
        # todo maybe batch the evaluations to allow large nudge samples on GPU
        original_prediction = outcome_model(policy_treatments, policy_covariates)
        nudged_predictions = outcome_model(nudged_treatments, policy_covariates)
        effect_predictions = nudged_predictions - original_prediction
        original_propensity = propensity_model(policy_treatments, policy_covariates)
        nudged_propensity = propensity_model(nudged_treatments, policy_covariates)
        original_logit = tr.sum(original_propensity * nudges, dim=-1)
        nudged_logit = tr.sum(nudged_propensity * nudges, dim=-1)
        propensity_log_ratio = (original_logit - nudged_logit) / (2*temperature)
        riesz = tr.exp(propensity_log_ratio) - 1
        pseudo_outcomes = riesz * (policy_outcomes - original_prediction) + effect_predictions # units x nudges
        split_pseudo_outcomes.append(pseudo_outcomes)
        split_effect_predictions.append(effect_predictions) # for a more naive version
        split_covariates.append(policy_covariates[:, 0, :])
        outcome_model.to(device)
        propensity_model.to(device)
      split_pseudo_outcomes = tr.cat(split_pseudo_outcomes, dim=0).T # nudges x all units
      split_effect_predictions = tr.cat(split_effect_predictions, dim=0).T
      split_covariates = tr.cat(split_covariates, dim=0) # all units x n_covariates
      policy, _ = learn_policy(split_pseudo_outcomes.detach().numpy())
      policy_nudges = tr.tensor(policy).T @ nudges / nudge_sample_size # all units x n_treatments
      direct_policy, _ = learn_policy(split_effect_predictions.detach().numpy())
      direct_policy_nudges = tr.tensor(direct_policy).T @ nudges / nudge_sample_size
      mode_indices = tr.argmax(tr.tensor(policy), dim=0)
      #policy_nudges = nudges[mode_indices, :] # get modes instead. units x n_treatments
      policy_effect = split_pseudo_outcomes[mode_indices, :].diagonal().mean()
      iterator.set_postfix(policy_effect_estimate=policy_effect.item())
      regression_covariates.append(split_covariates.to(device))
      regression_policies.append(policy_nudges.to(device))
      regression_direct_policies.append(direct_policy_nudges.to(device))
  regression_covariates = tr.cat(regression_covariates, dim=0) # these are also reshuffled basically
  regression_policies = tr.cat(regression_policies, dim=0)
  regression_direct_policies = tr.cat(regression_direct_policies, dim=0)
  ### generalize policy
  policy_model = NeuralModel(n_inputs=n_covariates, n_outputs=n_treatments).to(device)
  criterion = nn.MSELoss()
  optimizer = opt.Adam(policy_model.parameters(),
    lr=policy_learning_rate, weight_decay=weight_decay)
  insample_train = np.random.choice(range(len(insample)), size=int(0.8*len(insample)), replace=False)
  insample_test = np.setdiff1d(range(len(insample)), insample_train)
  valid_covariates = regression_covariates[insample_test, :]
  valid_policies = regression_policies[insample_test, :]
  with trange(n_epochs, desc="Policy Model") as iterator:
    for epoch in iterator:
      epoch_train_loss = []
      for batch_index in range(0, len(insample_train), batch_size):
        optimizer.zero_grad()
        batch_covariates = regression_covariates[insample_train[batch_index : batch_index+batch_size], :]
        batch_policies = regression_policies[insample_train[batch_index : batch_index+batch_size], :]
        output = policy_model(batch_covariates)
        train_loss = criterion(output, batch_policies)
        epoch_train_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
      valid_output = policy_model(valid_covariates)
      epoch_valid_loss = criterion(valid_output, valid_policies).item()
      iterator.set_postfix(train_loss=np.mean(epoch_train_loss), valid_loss=np.mean(epoch_valid_loss))
  policy_model.eval()
  ## direct model
  direct_policy_model = NeuralModel(n_inputs=n_covariates, n_outputs=n_treatments).to(device)
  criterion = nn.MSELoss()
  optimizer = opt.Adam(direct_policy_model.parameters(),
    lr=policy_learning_rate, weight_decay=weight_decay)
  valid_covariates = regression_covariates[insample_test, :]
  valid_policies = regression_direct_policies[insample_test, :]
  with trange(n_epochs, desc="Direct Policy Model") as iterator:
    for epoch in iterator:
      epoch_train_loss = []
      for batch_index in range(0, len(insample_train), batch_size):
        optimizer.zero_grad()
        batch_covariates = regression_covariates[insample_train[batch_index : batch_index+batch_size], :]
        batch_policies = regression_direct_policies[insample_train[batch_index : batch_index+batch_size], :]
        output = policy_model(batch_covariates)
        train_loss = criterion(output, batch_policies)
        epoch_train_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
      valid_output = direct_policy_model(valid_covariates)
      epoch_valid_loss = criterion(valid_output, valid_policies).item()
      iterator.set_postfix(train_loss=np.mean(epoch_train_loss), valid_loss=np.mean(epoch_valid_loss))
  direct_policy_model.eval()
  ## naive model
  naive_policy_model = NeuralModel(n_inputs=n_covariates, n_outputs=n_treatments).to(device)
  # averaging over outcome models like an ensemble seems better than partitioning
  criterion = lambda t, x: sum(-tr.mean(outcome_models[k](t, x)) for k in range(n_splits))
  optimizer = opt.Adam(naive_policy_model.parameters(),
    lr=naive_policy_learning_rate, weight_decay=weight_decay)
  valid_treatments = treatments[insample[insample_test], :]
  valid_covariates = covariates[insample[insample_test], :]
  with trange(n_epochs, desc="Naive Policy Model") as iterator:
    for epoch in iterator:
      epoch_train_loss = []
      for batch_index in range(0, len(insample_train), batch_size):
        optimizer.zero_grad()
        indices = insample[insample_train[batch_index : batch_index+batch_size]]
        batch_treatments = treatments[indices, :]
        batch_covariates = covariates[indices, :]
        output = naive_policy_model(batch_covariates)
        train_loss = criterion(batch_treatments+output, batch_covariates)
        epoch_train_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
      valid_output = naive_policy_model(valid_covariates)
      epoch_valid_loss = criterion(valid_treatments+valid_output, valid_covariates).item()
      iterator.set_postfix(train_loss=np.mean(epoch_train_loss), valid_loss=np.mean(epoch_valid_loss))
  naive_policy_model.eval()
  ## semi-naive model
  semi_policy_model = NeuralModel(n_inputs=n_covariates, n_outputs=n_treatments).to(device)
  def criterion(partitions, nudges, outcomes, treatments, covariates):
    loss = tr.zeros([], device=device)
    for split in range(n_splits):
      outcome_model = outcome_models[split]
      propensity_model = propensity_models[split]
      temperature = temperatures[split]
      split_mask = (partitions == split)
      split_treatments = treatments[split_mask, :]
      split_covariates = covariates[split_mask, :]
      split_outcomes = outcomes[split_mask]
      split_nudges = nudges[split_mask, :]
      nudged_treatments = split_nudges + split_treatments
      original_prediction = outcome_model(split_treatments, split_covariates)
      nudged_predictions = outcome_model(nudged_treatments, split_covariates)
      effect_predictions = nudged_predictions - original_prediction
      original_propensity = propensity_model(split_treatments, split_covariates)
      nudged_propensity = propensity_model(nudged_treatments, split_covariates)
      original_logit = tr.sum(original_propensity * split_nudges, dim=-1)
      nudged_logit = tr.sum(nudged_propensity * split_nudges, dim=-1)
      propensity_log_ratio = (original_logit - nudged_logit) / (2*temperature)
      riesz = tr.exp(propensity_log_ratio) - 1
      pseudo_outcome = riesz * (split_outcomes - original_prediction) + effect_predictions
      loss += -tr.mean(pseudo_outcome)
    return loss
  optimizer = opt.Adam(semi_policy_model.parameters(),
    lr=naive_policy_learning_rate, weight_decay=weight_decay)
  valid_partitions = partitions[insample_test]
  valid_treatments = treatments[insample[insample_test], :]
  valid_covariates = covariates[insample[insample_test], :]
  valid_outcomes = outcomes[insample[insample_test]]
  with trange(n_epochs, desc="Semi-naive Policy Model") as iterator:
    for epoch in iterator:
      epoch_train_loss = []
      for batch_index in range(0, len(insample_train), batch_size):
        optimizer.zero_grad()
        indices = insample[ insample_train[batch_index : batch_index+batch_size] ]
        batch_partitions = partitions[ insample_train[batch_index : batch_index+batch_size] ]
        batch_treatments = treatments[indices, :]
        batch_covariates = covariates[indices, :]
        batch_outcomes = outcomes[indices]
        output = semi_policy_model(batch_covariates)
        train_loss = criterion(batch_partitions, output, batch_outcomes, batch_treatments, batch_covariates)
        epoch_train_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
      valid_output = semi_policy_model(valid_covariates)
      epoch_valid_loss = criterion(valid_partitions, valid_output,
        valid_outcomes, valid_treatments, valid_covariates).item()
      iterator.set_postfix(train_loss=np.mean(epoch_train_loss), valid_loss=np.mean(epoch_valid_loss))
  semi_policy_model.eval()
  ### evaluate policy
  out_treatments = treatments[outsample, :]
  out_covariates = covariates[outsample, :]
  prescriptions = policy_model(out_covariates).detach()
  nudged_treatments = out_treatments + prescriptions
  direct_prescriptions = direct_policy_model(out_covariates).detach()
  direct_treatments = out_treatments + direct_prescriptions
  naive_prescriptions = naive_policy_model(out_covariates).detach()
  naive_treatments = out_treatments + naive_prescriptions
  semi_prescriptions = semi_policy_model(out_covariates).detach()
  semi_treatments = out_treatments + semi_prescriptions
  nudged_outcomes = generate_outcomes(mixing, nudged_treatments, out_covariates, noise=0, device=device)
  direct_outcomes = generate_outcomes(mixing, direct_treatments, out_covariates, noise=0, device=device)
  naive_outcomes = generate_outcomes(mixing, naive_treatments, out_covariates, noise=0, device=device)
  semi_outcomes = generate_outcomes(mixing, semi_treatments, out_covariates, noise=0, device=device)
  original_outcomes = generate_outcomes(mixing, out_treatments, out_covariates, noise=0, device=device)
  return (
    (original_outcomes, out_treatments),
    (nudged_outcomes, nudged_treatments),
    (direct_outcomes, direct_treatments),
    (naive_outcomes, naive_treatments),
    (semi_outcomes, semi_treatments) )

# a more sophisticated baseline like proximal policy optimization would require other hyperparameters.