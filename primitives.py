import torch
import copy

def vector(*x):
    # NOTE: This must support both lists and vectors
    try:
        result = torch.stack(x)
    except:
        result = list(x)
    return result


def hashmap(*x):
    _keys = [key for key in x[0::2]]
    keys = []
    for key in _keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is torch.Tensor: key = float(key)
        keys.append(key)
    values = [value for value in x[1::2]]
    return dict(zip(keys, values))


def first(vec):
    return vec[0]


def second(vec):
    return vec[1]


def rest(vec):
    return vec[1:]


def last(vec):
    return vec[-1]


def append(vec, val):
    vec = torch.cat([vec, val[None]])
    return vec


def nth(idx, vec):
    return vec[idx]


def conj(vec, elem):
    return torch.cat([vec, elem])


def cons(elem, vec):
    return torch.cat([elem, vec])


def get(vec, idx):
    return vec[idx.int().item()]

def put(hashmap, idx, val):
    hashmap[idx.int().item()] = val
    return hashmap

def repmat(vec, *dims):
    return torch.tile(vec, [dim.int() for dim in dims])

def remove(vec, index):
    part_1 = vec[:index]
    part_2 = vec[index+1:]
    return conj(part_1, part_2)

# Primative function dictionary
# NOTE: You should complete this
primitives = {

    # Comparisons
    '<': torch.lt,
    '<=': torch.le,
    '>': torch.gt,
    '>=': torch.ge,
    '=': torch.eq,
    'and': torch.logical_and,
    'or': torch.logical_or,
    'not': torch.logical_not,
    'first': first,
    'second': second,
    'last': last,
    'rest': rest,
    'nth': nth,
    'conj': conj,
    'cons': cons,
    'get': get,
    'put': put,
    'append': append,
    'remove': remove,
    'mat-transpose': lambda mat: mat.T,
    'mat-repmat': repmat,
    'mat-add': torch.add,
    'mat-tanh': torch.tanh,
    # ...

    # Math
    '+': torch.add,
    '-': torch.sub,
    '*': torch.mul,
    '/': torch.div,
    'sqrt': torch.sqrt,
    # ...

    # Containers
    'vector': vector,
    'hash-map': hashmap,
    # ...

    # Matrices
    'mat-mul': torch.matmul,
    # ...

    # Distributions
    'normal': torch.distributions.Normal,
    'uniform-continuous': torch.distributions.Uniform,
    'beta': torch.distributions.Beta,
    'bernoulli': torch.distributions.Bernoulli,
    'exponential': torch.distributions.Exponential,
    'discrete': torch.distributions.Categorical,
    'gamma': torch.distributions.Gamma,
    'dirichlet': torch.distributions.Dirichlet,
    'flip': torch.distributions.Bernoulli
    # ...

}