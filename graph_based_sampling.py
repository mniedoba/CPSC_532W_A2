import copy
from graphlib import TopologicalSorter

import torch
import numpy as np

import time
import copy

# Project imports
from evaluation_based_sampling import Expression, eval_expression, EvaluationScheme, ExpressionType
from utils import log_sample

class Graph:
    def __init__(self, graph_json):
        self.json = graph_json
        # Convert to node: predecessor format from node: successor format.
        node_preds = {node: set() for node in self.json[1]['V']}
        self.node_successors = self.json[1]['A']
        for node, successors in self.node_successors.items():
            for successor in successors:
                node_preds[successor].add(node)

        # Sort the nodes in topological order and save off the ordered list of nodes.
        self.topological_sorter = TopologicalSorter(node_preds)
        self.ordered_nodes = [node for node in self.topological_sorter.static_order()]

        # Link functions are mappings from nodes to expressions.
        self.link_functions = {k: Expression(v) for k, v in self.json[1]['P'].items()}
        self.latents = []
        self.observed = []
        for node in self.ordered_nodes:
            if self.link_functions[node].type == ExpressionType.SAMPLE:
                self.latents.append(node)
            elif self.link_functions[node].type == ExpressionType.OBSERVE:
                self.observed.append(node)

        # Procedures don't seem to be used in the syntax at the moment?? Not really sure what is up with that.
        self.procedures = self.json[0] #Todo.

        # The return of the program is just an expression.
        self.return_expression = Expression(self.json[-1])


def evaluate_graph(graph, eval_scheme=EvaluationScheme.PRIOR, verbose=False):

    node_vals = {}
    sigma =[]

    # Loop through the topological ordering of nodes, updating the environment as we go.
    # TODO: Do I need to think about scope here?
    for node in graph.ordered_nodes:
        node_vals[node], sigma = eval_expression(graph.link_functions[node], sigma, node_vals, graph.procedures)

    # Evaluate the return expression with the sampled values.
    r_val, sigma = eval_expression(graph.return_expression, sigma, node_vals, graph.procedures)
    return r_val, sigma, node_vals


def compute_joint(graph, node, values):
    """Computes the joint probability of the node and its parents."""
    sigma = []
    log_prob = 0.
    for succ in graph.node_successors[node]:
        link_fnc = graph.link_functions[succ]
        if link_fnc.type == ExpressionType.SAMPLE:
            d1, sigma = eval_expression(link_fnc[0], sigma, values, graph.procedures)
            node_val = values[succ][0]
            log_prob += d1.log_prob(node_val)
        elif link_fnc.type == ExpressionType.OBSERVE:
            assert len(link_fnc.sub_expressions) == 2
            e1, e2 = link_fnc.sub_expressions
            d1, sigma = eval_expression(e1, sigma, values, graph.procedures)
            c2, sigma = eval_expression(e2, sigma, values, graph.procedures)
            log_prob += d1.log_prob(c2)
    return log_prob

def metropolis_within_gibbs(graph, mode, num_samples, wandb_name=None, tmax=None, burn_in=1000):
    assert mode == 'graph'

    return_samples = []
    sigma = []
    node_values = {}
    for node in graph.ordered_nodes:
        node_val, sigma = eval_expression(graph.link_functions[node], sigma, node_values, graph.procedures)
        node_values[node] = [node_val]

    done = False
    i = 0
    start_time = time.time()
    while not done:
        new_node_values = copy.deepcopy(node_values)
        for node in graph.ordered_nodes:
            if node in graph.latents:
                node_prior, sigma = eval_expression(graph.link_functions[node][0], sigma, new_node_values, graph.procedures)
                old_joint_log_prob = compute_joint(graph, node, new_node_values)
                acc_prob = -1.
                while np.random.random() > acc_prob:
                    x_prime = node_prior.sample()
                    new_node_values[node][0] = x_prime
                    new_joint_log_prob = compute_joint(graph, node, new_node_values)
                    acc_log_prob = new_joint_log_prob - old_joint_log_prob
                    acc_prob = min(1, torch.exp(acc_log_prob).item())
        if i >= burn_in:
            r_val, sigma = eval_expression(graph.return_expression, sigma, new_node_values, graph.procedures)
            return_samples.append(r_val)
            if wandb_name is not None:
                log_sample(r_val, i, wandb_name)
            node_values = new_node_values
        i += 1
        done = (time.time() - start_time >= tmax) or len(return_samples) == num_samples
    num_samples = len(return_samples)
    return_samples = torch.stack(return_samples).float()
    weights = torch.ones(num_samples) / num_samples

    return return_samples, weights


def get_potential(graph, sample):
    sigma = []
    for v in sample.values():
        if not v[-1].requires_grad:
            v[-1].requires_grad = True
        v[-1].grad = torch.zeros_like(v[-1])
    log_prob = 0
    for node in graph.ordered_nodes:
        link_fnc = graph.link_functions[node]
        if link_fnc.type == ExpressionType.SAMPLE:
            d1, sigma = eval_expression(link_fnc[0], sigma, sample, graph.procedures)
            node_val = sample[node][-1]
            # print(f'Sampling {node_val} log_prob {d1.log_prob(node_val)}')
            log_prob += d1.log_prob(node_val)
        elif link_fnc.type == ExpressionType.OBSERVE:
            assert len(link_fnc.sub_expressions) == 2
            e1, e2 = link_fnc.sub_expressions
            d1, sigma = eval_expression(e1, sigma, sample, graph.procedures)
            c2, sigma = eval_expression(e2, sigma, sample, graph.procedures)
            # print(f'Observing {c2}, log_prob {d1.log_prob(c2)}')
            log_prob += d1.log_prob(c2)

    potential = -log_prob
    # print(sample), potential
    return potential

def leapfrog_integrate(x0, R0, T, epsilon, graph):
    get_potential(graph, x0).backward()
    R = {}
    for k in R0:
        R[k] = R0[k] + 0.5 - epsilon * x0[k][-1].grad
    X = x0
    for t in range(T):
        R_new, X_new = {}, {}
        for k in X:
            X_new[k] = [(X[k][-1] + epsilon * R[k]).detach()]
        get_potential(graph, X_new).backward()
        for k in R:
            R_new[k] = R[k] - 0.5 * epsilon * X_new[k][-1].grad
        X, R = X_new, R_new
    return X, R


def HMC(graph, mode, num_samples, wandb_name, tmax=None):

    start_time = time.time()
    x0 = {}
    return_samples = []
    sigma = []
    for node in graph.ordered_nodes:
        node_val, sigma = eval_expression(graph.link_functions[node], sigma, x0, graph.procedures)
        if node_val is not None:
            x0[node] = [node_val]
    x = x0

    T = 50
    epsilon = 1E-2

    done = False

    while not done:
        R = {k: torch.randn_like(v[-1]) for k,v in x.items() }
        x_prime, R_prime = leapfrog_integrate(x, R, T, epsilon, graph)
        U = get_potential(graph, x_prime)
        U_old = get_potential(graph, x)
        K = 0
        K_old = 0
        for k in R:
            r_prime = R_prime[k]
            r = R[k]
            if r.dim() == 0:
                K_old += 0.5 * r*r
                K += 0.5 * r_prime * r_prime
            else:
                K += 0.5 * torch.dot(r_prime, r_prime)
                K_old += 0.5 * torch.dot(r, r)
        H =  U + K
        H_old = U_old + K_old
        if np.random.random() < torch.exp(H_old - H):
            x = x_prime
        r_sample, _ = eval_expression(graph.return_expression,
                                      sigma, x, graph.procedures)
        if wandb_name is not None:
            log_sample(r_sample, len(return_samples), wandb_name)
        return_samples.append(r_sample)

        if tmax is not None:
            done = (time.time() - start_time) >= tmax
        else:
            done = len(return_samples) == num_samples

    num_samples = len(return_samples)
    weights = torch.ones(num_samples) / num_samples
    return_samples = torch.stack(return_samples)
    return return_samples, weights