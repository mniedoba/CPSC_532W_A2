# Standard imports
import torch as tc
from time import time

# Project imports
from evaluation_based_sampling import evaluate_program, EvaluationScheme
from graph_based_sampling import evaluate_graph, metropolis_within_gibbs, HMC
from utils import log_sample

def flatten_sample(sample):
    if type(sample) is list: # NOTE: Nasty hack for the output from program 4 of homework 2
        flat_sample = tc.concat([element.flatten() for element in sample])
    else:
        flat_sample = sample
    return flat_sample


def get_sample(ast_or_graph, mode, verbose=False):
    if mode == 'desugar':
        ret, sig, _ = evaluate_program(ast_or_graph, verbose=verbose)
    elif mode == 'graph':
        ret, sig, _ = evaluate_graph(ast_or_graph, verbose=verbose)
    else:
        raise ValueError('Mode not recognised')
    ret = flatten_sample(ret)
    return ret, sig


def prior_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a FOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, _ = get_sample(ast_or_graph, mode, verbose)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and time() > max_time: break
    weights = tc.ones(num_samples)
    weights /= weights.sum()
    return samples, weights


def importance_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    """Generate a set of samples and associated weights from the posterior of a FOPPL program."""
    samples, weights = [], []
    if (tmax is not None):
        max_time = time() + tmax
    done = False
    while not done:
        if mode == 'desugar':
            sample, sigma, _ = evaluate_program(ast_or_graph, eval_scheme=EvaluationScheme.IS, verbose=verbose)
            samples.append(sample)
            weights.append(tc.exp(sigma['log_w']))
        else:
            samples, sigma, _ = evaluate_graph(ast_or_graph, eval_scheme=EvaluationScheme.IS, verbose=verbose)
        done = len(samples) == num_samples or time() >= max_time
    print(samples[0].shape)
    sample_tensor = tc.stack(samples)
    weight_tensor = tc.tensor(weights)
    weight_tensor /= weight_tensor.sum()
    while len(weight_tensor.shape) < len(sample_tensor.shape):
        weight_tensor = weight_tensor[..., None]

    print(f'Posterior mean: {(sample_tensor * weight_tensor).sum(dim=0)}')

    return sample_tensor, weight_tensor



def sample(ast_or_graph, mode, eval_scheme, num_samples, tmax=None, wandb_name=None, verbose=False):
    match eval_scheme:
        case EvaluationScheme.PRIOR:
            return prior_samples(ast_or_graph, mode, num_samples, tmax, wandb_name, verbose)
        case EvaluationScheme.IS:
            return importance_samples(ast_or_graph, mode, num_samples, tmax, wandb_name, verbose)
        case EvaluationScheme.METROPOLIS_GIBBS:
            return metropolis_within_gibbs(ast_or_graph, mode, num_samples, wandb_name, tmax,)
        case EvaluationScheme.HMC:
            return HMC(ast_or_graph, mode, num_samples, wandb_name, tmax,)
        case other:
            raise ValueError(f'Unknown eval scheme {eval_scheme}')