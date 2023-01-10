import numpy as np
from copy import deepcopy
from scipy.stats import norm
from scipy.optimize import fminbound
from random import randint, random
#from mtsoo import *

# new corssover operators
def x_single_point(p1, p2, _sbxdi=None, _pc=None):
  #print('single_point_x')
  D = p1.shape[0]
  # Randomly generating one crossover point
  crossover_point = randint(1, D-2)

  c1 = deepcopy(p1)
  c2 = deepcopy(p2)
  # Crossing over from randomly generated point
  c1[crossover_point:] = p2[crossover_point:]
  c2[crossover_point:] = p1[crossover_point:]

  return c1, c2


def x_two_point(p1, p2, _sbxdi=None, _pc=None):
  #print('two_point_x')
  D = p1.shape[0]
  # Randomly generating two crossover point
  cross_point_num = int(2)
  cross_range = np.arange(1,D-1)
  cross_point_idx = np.random.choice(cross_range, size=cross_point_num, replace=False)
  cross_point_idx.sort()
  point_1 = cross_point_idx[0]
  point_2 = cross_point_idx[1]

  c1 = deepcopy(p1)
  c2 = deepcopy(p2)
  # Crossing over between randomly generated points
  c1[point_1:point_2] = p2[point_1:point_2]
  c2[point_1:point_2] = p1[point_1:point_2]

  return c1, c2

def x_uniform_point(p1, p2, _sbxdi=None, _pc=None):
  #print('uniform_point_x')
  D = p1.shape[0]
  c1 = deepcopy(p1)
  c2 = deepcopy(p2)

  probability_of_swap = 0.5
  for idx, gene in enumerate(p1):
    r = random()
    if r > probability_of_swap:
      c1[idx] = p2[idx]
      c2[idx] = p1[idx]
  return c1, c2

def x_arithmetic(p1, p2, _sbxdi=None, _pc=None):
  # float
  if _pc == None:
    pc = 0.6
  else:
    pc = _pc

  D = p1.shape[0]
  c1 = deepcopy(p1)
  c2 = deepcopy(p2)
  randVar= np.random.rand()
  if randVar<pc:
    w=np.random.rand()
    for i in range(D):
        c1[i] = p1[i]*w + p2[i]*(1-w)
        c2[i] = p1[i]*(1-w) + p2[i]*w
    return c1, c2
  else:
    return p1, p2

def x_geometrical(p1, p2, _sbxdi=None, _pc=None):
  # float    
  D = p1.shape[0]
  c1 = deepcopy(p1)
  c2 = deepcopy(p2)

  for i in range(D):
    c1[i] = np.sqrt(p1[i]*p2[i])  # == np.power(p1[i], 0.5)*np.power(p2[i], 0.5)
    randVar= np.random.rand()
    c2[i] = np.power(p1[i], randVar)*np.power(p2[i], 1-randVar)
  return c1, c2

def x_BLX_alpha(p1, p2, _sbxdi=None, _pc=None):
  # float    
  D = p1.shape[0]
  c1 = deepcopy(p1)
  c2 = deepcopy(p2)

  alpha = 0.5
  randVar= np.random.rand()
  gamma = (1+2*alpha)*randVar-alpha

  for i in range(D):
      c1[i] = (1-gamma)*p1[i] + gamma*p2[i]
      c2[i] = (1-gamma)*p2[i] + gamma*p1[i]
  return c1, c2

def x_cut_splice(p1, p2, _sbxdi=None, _pc=None):
  #print('cut_splice_x')
  p1 = list(p1)
  p2 = list(p2)
  #D = p1.shape[0]
  D = len(p1)
  # Randomly generating two crossover point
  cross_point_num = int(2)
  cross_range = np.arange(1,D-1)
  cross_point_idx = np.random.choice(cross_range, size=cross_point_num, replace=False)
  cross_point_idx.sort()
  point_1 = cross_point_idx[0]
  point_2 = cross_point_idx[1]

  c1 = deepcopy(p1)
  c2 = deepcopy(p2)
  # Crossing over between randomly generated points
  c1[point_1:] = p2[point_2:]
  c2[point_2:] = p1[point_1:]

  # To ensure list in not smaller than D
  var1 = list((randint(1, D) for x in range(len(c1), D)))
  c1.extend(var1)
  var2 = list((randint(1, D) for x in range(len(c2), D)))
  c2.extend(var2)

  # To ensure list is not bigger than D
  del c1[D:]
  del c2[D:]

  c1 = np.array(c1)
  c2 = np.array(c2)
  return c1, c2

def xx(p1, p2):
  p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
  p2 = np.array([11, 22, 33, 44, 55, 66, 77, 88, 99])
  D = p1.shape[0]
  c1 = deepcopy(p1)
  c2 = deepcopy(p2)

  probability_of_swap = 0.5
  for idx, gene in enumerate(p1):
    r = random()
    print(r)
    if r > probability_of_swap:
      c1[idx] = p2[idx]
      c2[idx] = p1[idx]
  print(f'c1: {c1}')
  print(f'c2: {c2}')
  input()
  return c1, c2

# EVOLUTIONARY OPERATORS
def x_sbx(p1, p2, _sbxdi=None, _pc=None):
  #float
  if _sbxdi == None:
    sbxdi = 10
  else:
    sbxdi = _sbxdi
  D = p1.shape[0]
  cf = np.empty([D])
  #print(cf)
  u = np.random.rand(D)       
  #print(u) 

  cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (sbxdi + 1)))
  cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (sbxdi + 1)))

  c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
  c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

  c1 = np.clip(c1, 0, 1)
  c2 = np.clip(c2, 0, 1)
  #print(f'c1: {c1}')
  #print(f'c2: {c2}')
  #input()

  return c1, c2

def mutate(p, pmdi):
  mp = float(1. / p.shape[0])
  u = np.random.uniform(size=[p.shape[0]])
  r = np.random.uniform(size=[p.shape[0]])
  tmp = np.copy(p)
  for i in range(p.shape[0]):
    if r[i] < mp:
      if u[i] < 0.5:
        delta = (2*u[i]) ** (1/(1+pmdi)) - 1
        tmp[i] = p[i] + delta * p[i]
      else:
        delta = 1 - (2 * (1 - u[i])) ** (1/(1+pmdi))
        tmp[i] = p[i] + delta * (1 - p[i])
  tmp = np.clip(tmp, 0, 1)
  return tmp

def variable_swap(p1, p2, probswap):
  #p1 = np.array([1, 2, 3, 4, 5])
  #p2 = np.array([6, 7, 8, 9, 0])
  D = p1.shape[0]
  swap_indicator = np.random.rand(D) <= probswap
  c1, c2 = p1.copy(), p2.copy()
  c1[np.where(swap_indicator)] = p2[np.where(swap_indicator)]
  c2[np.where(swap_indicator)] = p1[np.where(swap_indicator)]
  #print(f'c1: {c1}')
  #print(f'c2: {c2}')
  #input()
  return c1, c2

# MULTIFACTORIAL EVOLUTIONARY HELPER FUNCTIONS
def find_relative(population, skill_factor, sf, N):
  return population[np.random.choice(np.where(skill_factor[:N] == sf)[0])]

def calculate_scalar_fitness(factorial_cost):
  return  1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)

def calculate_scalar_fitness_single(factorial_cost):

  result = 1 / (np.argsort(factorial_cost) + 1)

  return result

# MULTIFACTORIAL EVOLUTIONARY WITH TRANSFER PARAMETER ESTIMATION HELPER FUNCTIONS
def get_subpops(population, skill_factor, N):
  K = len(set(skill_factor))
  subpops = []
  for k in range(K):
    idx = np.where(skill_factor == k)[0][:N//K]
    subpops.append(population[idx, :])
  return subpops

class Model:
  def __init__(self, mean, std, num_sample):
    self.mean        = mean
    self.std         = std
    self.num_sample  = num_sample

  def density(self, subpop):
    N, D = subpop.shape
    prob = np.ones([N])
    for d in range(D):
      prob *= norm.pdf(subpop[:, d], loc=self.mean[d], scale=self.std[d])
    return prob

def log_likelihood(rmp, prob_matrix, K):
  posterior_matrix = deepcopy(prob_matrix)
  value = 0
  for k in range(2):
    for j in range(2):
      if k == j:
        posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * (1 - 0.5 * (K - 1) * rmp / float(K))
      else:
        posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * 0.5 * (K - 1) * rmp / float(K)
    value = value + np.sum(-np.log(np.sum(posterior_matrix[k], axis=1)))
  return value

def learn_models(subpops):
  K = len(subpops)
  D = subpops[0].shape[1]
  models = []
  for k in range(K):
    subpop            = subpops[k]
    num_sample        = len(subpop)
    num_random_sample = int(np.floor(0.1 * num_sample))
    rand_pop          = np.random.rand(num_random_sample, D)
    mean              = np.mean(np.concatenate([subpop, rand_pop]), axis=0)
    std               = np.std(np.concatenate([subpop, rand_pop]), axis=0)
    models.append(Model(mean, std, num_sample))
  return models

def learn_rmp(subpops, D):
  K          = len(subpops)
  rmp_matrix = np.eye(K)
  #print(f'K {K}')
  #print(f'rmp_matrix\n{rmp_matrix}')
  #input()
  models = learn_models(subpops)

  for k in range(K - 1):
    for j in range(k + 1, K):
      probmatrix = [np.ones([models[k].num_sample, 2]), 
                    np.ones([models[j].num_sample, 2])]
      probmatrix[0][:, 0] = models[k].density(subpops[k])
      probmatrix[0][:, 1] = models[j].density(subpops[k])
      probmatrix[1][:, 0] = models[k].density(subpops[j])
      probmatrix[1][:, 1] = models[j].density(subpops[j])

      rmp = fminbound(lambda rmp: log_likelihood(rmp, probmatrix, K), 0, 1)
      rmp += np.random.randn() * 0.01
      rmp = np.clip(rmp, 0, 1)
      rmp_matrix[k, j] = rmp
      rmp_matrix[j, k] = rmp

  return rmp_matrix

# OPTIMIZATION RESULT HELPERS
def get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, sf):
  # select individuals from task sf
  idx                = np.where(skill_factor == sf)[0]
  subpop             = population[idx]
  sub_factorial_cost = factorial_cost[idx]
  sub_scalar_fitness = scalar_fitness[idx]

  # select best individual
  idx = np.argmax(sub_scalar_fitness)
  x = subpop[idx]
  fun = sub_factorial_cost[idx, sf]
  return x, fun

def get_best_individual_single(population, factorial_cost, scalar_fitness, skill_factor):
  # select individuals from task sf
  

  # select best individual

  idx = np.argmax(scalar_fitness)
  x = population[idx]
  fun = factorial_cost[idx]
  return x, fun