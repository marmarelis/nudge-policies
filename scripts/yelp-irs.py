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
import os

if os.path.isfile("../style.mplstyle"):
  from matplotlib import pyplot as plt
  plt.ion()
  plt.style.use("../style.mplstyle")


df = pd.read_csv("data/yelp-irs-experiment.csv")

lat = 1.0 * (df['latitude'] -  df['latitude'].mean())
lon = 1.0 * (df['longitude'] -  df['longitude'].mean())
#geo_mask = (lat > -5) & (lat < 5) & (lon > -5) & (lon < 5) # for CA dataset: only roughly LA area
geo_mask = ~np.isnan(lat) # vacuous mask for now

clustering = KMeans(11, n_init="auto", random_state=0) # the 11 metropolitan areas
cluster_id = clustering.fit_predict(np.vstack([lon, lat]).T)
centroids = clustering.cluster_centers_[cluster_id, :] # not used for now

covariates = np.vstack([
    df['umap1'], df['umap2'], df['umap3'], df['umap4'], df['umap5'],
    df['review_count'], df['review_stars'], df['is_open'] ]).T
treatments = np.vstack([lat, lon]).T

knn = NearestNeighbors(n_neighbors=3).fit(treatments)
composite_treatments = np.reshape(treatments[knn.kneighbors(treatments)[1], :], (-1, 6))

def compute_outcome(covariates, treatments, expectation=False, interference=False):
  _, knn_indices = knn.kneighbors(treatments)
  first_bracket = df['first_income_bracket'][knn_indices[:, 0]]
  second_bracket = df['second_income_bracket'][knn_indices[:, 0]]
  if interference: # graph stays fixed, but brackets change and interference mixes them
    first_bracket = np.mean(first_bracket[knn_indices], axis=1)
    second_bracket = np.mean(second_bracket[knn_indices], axis=1)
  umap = covariates[:, 0:5]
  umap_norm = np.linalg.norm(umap, ord=2, axis=1)
  # add a bit of randomness to study conditional expectations
  review_noise = np.random.exponential(scale=10, size=covariates.shape[0])
  review_count = np.sqrt(covariates[:, 5]) + (review_noise if not expectation else 10)
  review_stars = covariates[:, 6]
  is_open = covariates[:, 7]
  irs_response = 10*first_bracket + is_open*second_bracket
  outcome = irs_response * review_count * review_stars / umap_norm
  return np.array(outcome) / 1e4

np.random.seed(0)
outcome = compute_outcome(covariates, treatments)

np.random.seed(0)
unit_indices = np.arange(outcome.size)[geo_mask]
n_units = np.sum(geo_mask)
np.random.shuffle(unit_indices)
train_indices = unit_indices[:(n_units // 2)]
test_indices = unit_indices[(n_units // 2):]

nudge_radius = 1 # 1 unit is about 50-60 miles


import xgboost as xgb
from sklearn.model_selection import GridSearchCV

train_input = np.concatenate([covariates, treatments], axis=1)[train_indices, :]
train_output = outcome[train_indices]
test_input = np.concatenate([covariates, treatments], axis=1)[test_indices, :]
test_output = outcome[test_indices]

def train_output_predictor(n_jobs=1):
  # taken from https://xgboost.readthedocs.io/en/stable/python/examples/sklearn_parallel.html
  xgb_model = xgb.XGBRegressor(n_jobs=1, random_state=0)
  xgb_fit = GridSearchCV(xgb_model,
    {'max_depth': range(3,10), 'n_estimators': [4_000, 8_000, 12_000, 16_000, 20_000]},
    cv=5, verbose=3, n_jobs=n_jobs)
  xgb_fit.fit(train_input, train_output)
  xgb_model = xgb_fit.best_estimator_
  #xgb_model = xgb.XGBRegressor(**xgb_fit.best_params_)
  #xgb_model.fit(train_input, train_output)
  return xgb_model


import torch as tr
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt

class PropensityModel(nn.Module):
  def __init__(self, width, depth):
    super().__init__()
    self.first = nn.Linear(10, width if depth>0 else 2)
    self.middle = nn.ModuleList([
      nn.Linear(width, width) for _ in range(depth-1) ])
    self.last = nn.Linear(width, 2) if depth>0 else None
    self.dropout = nn.Dropout(p=0.05)

  def forward(self, input):
    if self.last is None:
      return self.first(input)
    inner = self.dropout( F.silu(self.first(input)) )
    for layer in self.middle:
      inner = self.dropout( F.silu(layer(inner)) )
    return self.last(inner)

def train_propensity_predictor(n_epochs=1_000_000,
    learning_rate=1e-5, weight_decay=1e-5, width=256, depth=3, device='cpu'):
  tr.manual_seed(0)
  input = tr.tensor(train_input, dtype=tr.float).to(device)
  n_units = input.shape[0]
  train_indices = range(0, int(0.9 * n_units))
  valid_indices = range(int(0.9 * n_units), n_units)
  model = PropensityModel(width, depth).to(device)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = opt.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  train_losses, valid_losses = [], [] 
  with trange(n_epochs) as iterator:
    for epoch in iterator:
      optimizer.zero_grad()
      label = tr.randint(2, (n_units, 1), dtype=tr.float, device=device)
      noise = nudge_radius * tr.randn((n_units, 2), device=device)
      noisy_treatment = input[:, -2:]  +  noise * label
      noisy_input = tr.concatenate([input[:, :-2], noisy_treatment], dim=1)
      output = model(noisy_input)
      logit = tr.sum(output * noise, dim=1)
      train_loss = criterion(logit[train_indices], label[train_indices, 0])
      valid_loss = criterion(logit[valid_indices], label[valid_indices, 0])
      train_losses.append(train_loss.item())
      valid_losses.append(valid_loss.item())
      iterator.set_postfix(train_loss=train_losses[-1], valid_loss=valid_losses[-1])
      train_loss.backward()
      optimizer.step()
  return model.to('cpu'), train_losses, valid_losses

def train_score_predictor(n_epochs=1_000_000,
    learning_rate=1e-5, weight_decay=1e-5, width=256, depth=3, device='cpu'):
  tr.manual_seed(0)
  input = tr.tensor(train_input, dtype=tr.float).to(device)
  n_units = input.shape[0]
  train_indices = range(0, int(0.9 * n_units))
  valid_indices = range(int(0.9 * n_units), n_units)
  model = PropensityModel(width, depth).to(device)
  def criterion(output, noise):
    score = -noise / nudge_radius**2 # Gaussian score with mean zero
    return tr.mean( (score - output)**2 )
  optimizer = opt.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  train_losses, valid_losses = [], [] 
  with trange(n_epochs) as iterator:
    for epoch in iterator:
      optimizer.zero_grad()
      noise = nudge_radius * tr.randn((n_units, 2), device=device)
      noisy_treatment = input[:, -2:] + noise
      noisy_input = tr.concatenate([input[:, :-2], noisy_treatment], dim=1)
      output = model(noisy_input)
      train_loss = criterion(output[train_indices, :], noise[train_indices, :])
      valid_loss = criterion(output[valid_indices, :], noise[valid_indices, :])
      train_losses.append(train_loss.item())
      valid_losses.append(valid_loss.item())
      iterator.set_postfix(train_loss=train_losses[-1], valid_loss=valid_losses[-1])
      train_loss.backward()
      optimizer.step()
  return model.to('cpu'), train_losses, valid_losses

# study temperature-scaling calibration (see e.g. Guo 2017) with our noise-dependent classifier
def evaluate_pseudo_outcomes(outcome_model, propensity_model, n_rep=1, riesz_ceiling=np.inf): # <- scope creep...
  nudged = np.copy(test_input)
  nudge = nudge_radius * np.random.randn(test_input.shape[0], 2)
  nudged[:, -2:] += nudge
  nudged_outcome = np.vstack([ compute_outcome(nudged[:, :-2], nudged[:, -2:]) for _ in range(n_rep) ])
  original_outcome = np.vstack([ compute_outcome(test_input[:, :-2], test_input[:, -2:]) for _ in range(n_rep) ])
  effect = nudged_outcome - original_outcome
  nudged_expectation = compute_outcome(nudged[:, :-2], nudged[:, -2:], expectation=True)
  original_expectation = compute_outcome(test_input[:, :-2], test_input[:, -2:], expectation=True)
  effect_expectation = nudged_expectation - original_expectation
  original_prediction = outcome_model.predict(test_input)
  nudged_prediction = outcome_model.predict(nudged)
  effect_prediction = nudged_prediction - original_prediction
  original_propensity_input = tr.tensor(test_input, dtype=tr.float)
  nudged_propensity_input = tr.tensor(nudged, dtype=tr.float)
  original_propensity = propensity_model(original_propensity_input)
  nudged_propensity = propensity_model(nudged_propensity_input)
  original_logit = tr.sum(original_propensity * tr.tensor(nudge), dim=1).double() # <- make float64
  nudged_logit = tr.sum(nudged_propensity * tr.tensor(nudge), dim=1).double()
  eval_temp = lambda t: -tr.mean(
    tr.log(tr.sigmoid(nudged_logit/t)) + tr.log(1-tr.sigmoid(original_logit/t)) )
  temperature = 1.0
  temp_delta = 0.02
  best_loss = eval_temp(temperature)
  for _ in range(10_000):
    current_loss = eval_temp(temperature+temp_delta)
    if current_loss > best_loss:
      break
    best_loss = current_loss
    temperature += temp_delta
  #losses = [ eval_temp(t).item() for t in np.linspace(1,5,100) ]
  print(f"Optimized temperature to {temperature}.")
  propensity_log_ratio = (original_logit - nudged_logit) / (2*temperature) # averaging two equivalent predictions? this part seems necessary..
  #riesz = tr.min( propensity_ratio-1, tr.full((), 1.0) ).detach().numpy()
  riesz = tr.min( tr.exp(propensity_log_ratio) - 1, tr.full((), riesz_ceiling) ).detach().numpy()
  # we could evaluate pseudo-outcome expectations wrt Y|T,X, but what we'd really need is (Y,T)|X. therefore, we just do empirical risk.
  pseudo_outcome = riesz * (original_outcome-original_prediction) + effect_prediction
  nudge_size = np.sqrt( np.sum( nudge ** 2, axis=1 ) )
  return effect, effect_prediction, pseudo_outcome, nudge_size


# set `matched=False` for unconstrained baseline
def learn_policy(effects, reg=0, matched=True): # nudges x units
  assert reg >= 0
  original_dtype = effects.dtype
  n_nudges, n_units = effects.shape
  length = n_nudges * n_units
  # units inner, nudges outer
  coefficients = effects.reshape(-1).astype(np.float32) - reg
  #if reg == 0: doesn't actually make sense to have sparse vector
  #  coefficients = csc_array(
  #    np.where(np.abs(coefficients) < sparse_threshold, 0, coefficients) )
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


def evaluate_policy(outcome_model, propensity_model, n_nudges, n_units, riesz_ceiling=np.inf, verbose=True):
  unit_indices = np.random.choice(np.arange(test_input.shape[0]), size=n_units, replace=False)
  nudged = np.copy(test_input[unit_indices, :])
  nudge = nudge_radius * np.random.randn(n_nudges, 1, 2)
  nudged = np.concatenate([
    np.tile(nudged[None, :, :-2], [n_nudges, 1, 1]),
    nudged[:, -2:] + nudge], axis=-1) # expand dimensions
  nudged_outcome = np.stack([ compute_outcome(nudged[i, :, :-2], nudged[i, :, -2:]) for i in range(n_nudges) ])
  original_outcome = compute_outcome(test_input[:, :-2], test_input[:, -2:])
  effect = nudged_outcome - original_outcome[unit_indices]
  nudged_expectation = np.stack([ compute_outcome(nudged[i, :, :-2], nudged[i, :, -2:], expectation=True) for i in range(n_nudges) ])
  original_expectation = compute_outcome(test_input[unit_indices, :-2], test_input[unit_indices, -2:], expectation=True)
  effect_expectation = nudged_expectation - original_expectation
  original_prediction = outcome_model.predict(test_input[unit_indices, :])
  nudged_prediction = np.stack([ outcome_model.predict(nudged[i, ...]) for i in range(n_nudges) ])
  effect_prediction = nudged_prediction - original_prediction
  original_propensity_input = tr.tensor(test_input[unit_indices, :], dtype=tr.float)
  nudged_propensity_input = tr.tensor(nudged, dtype=tr.float)
  original_propensity = propensity_model(original_propensity_input)
  nudged_propensity = propensity_model(nudged_propensity_input)
  original_logit = tr.sum(original_propensity * tr.tensor(nudge), dim=-1).double() # <- make float64
  nudged_logit = tr.sum(nudged_propensity * tr.tensor(nudge), dim=-1).double()
  eval_temp = lambda t: -tr.mean(
    tr.log(tr.sigmoid(nudged_logit/t)) + tr.log(1-tr.sigmoid(original_logit/t)) )
  temperature = 1.0
  temp_delta = 0.02
  best_loss = eval_temp(temperature)
  for _ in range(10_000):
    current_loss = eval_temp(temperature+temp_delta)
    if current_loss > best_loss:
      break
    best_loss = current_loss
    temperature += temp_delta
  if verbose:
    print(f"Optimized temperature to {temperature}.")
  propensity_log_ratio = (original_logit - nudged_logit) / (2*temperature) # averaging two equivalent predictions? this part seems necessary..
  riesz = tr.min( tr.exp(propensity_log_ratio) - 1, tr.full((), riesz_ceiling) ).detach().numpy()
  # we could evaluate pseudo-outcome expectations wrt Y|T,X, but what we'd really need is (Y,T)|X. therefore, we just do empirical risk.
  pseudo_outcome = riesz * (original_outcome[unit_indices]-original_prediction) + effect_prediction
  ## POLICIES
  true_policy = learn_policy(effect, reg=0)
  naive_policy = learn_policy(effect_prediction, reg=0)
  naive_policy_effect = np.mean(effect * naive_policy[0], axis=0)
  estimated_policy = learn_policy(pseudo_outcome, reg=0)
  estimated_policy_effect = np.mean(effect * estimated_policy[0], axis=0)
  uncons_policy = learn_policy(pseudo_outcome, reg=0, matched=False)
  uncons_policy_effect = np.mean(effect * uncons_policy[0], axis=0)
  # more sophisticated importance weighting than this? take care to allow inflation of zeros...
  # right now it's rather conservative. I cut off what I can't afford.
  # note that this ablation is a bit strange and isn't reported in the paper.
  recons_policy = uncons_policy[0] * np.minimum(1,
    n_units / np.maximum(1e-10, np.sum(uncons_policy[0], axis=1, keepdims=True)) )
  recons_policy_est_effect = np.mean(pseudo_outcome * recons_policy)
  recons_policy_effect = np.mean(effect * recons_policy, axis=0)
  ## GENERALIZATIONS. not really used yet.
  naive_policy_nudges = naive_policy[0].T @ nudge[:, 0, :] / n_nudges # (units x 2)
  estimated_policy_nudges = estimated_policy[0].T @ nudge[:, 0, :] / n_nudges
  uncons_policy_nudges = uncons_policy[0].T @ nudge[:, 0, :] / n_nudges
  naive_gen = LinearRegression().fit(test_input[unit_indices, :5], naive_policy_nudges).predict(test_input[:, :5])
  naive_gen_effect = compute_outcome(test_input[:, :-2], naive_gen) - original_outcome
  estimated_gen = LinearRegression().fit(test_input[unit_indices, :5], estimated_policy_nudges).predict(test_input[:, :5])
  estimated_gen_effect = compute_outcome(test_input[:, :-2], estimated_gen) - original_outcome
  uncons_gen = LinearRegression().fit(test_input[unit_indices, :5], uncons_policy_nudges).predict(test_input[:, :5])
  uncons_gen_effect = compute_outcome(test_input[:, :-2], uncons_gen) - original_outcome
  return ( true_policy,
    estimated_policy + (estimated_policy_effect, estimated_gen_effect),
    naive_policy + (naive_policy_effect, naive_gen_effect),
    uncons_policy + (uncons_policy_effect, uncons_gen_effect),
    (recons_policy, recons_policy_est_effect, recons_policy_effect) )
