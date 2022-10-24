import torch
from torch.optim import Adam

from distributions import Normal, Gamma, Exponential, Beta, Dirichlet, Bernoulli, Categorical, DistributionCollection
from evaluation_based_sampling import ExpressionType, eval_expression


def get_variational_distribution_from_prior(graph):

    distributions = []
    latent_values = {}
    for node in graph.latents:
        prior_distribution, _ = eval_expression(graph.link_functions[node][0], [], latent_values, graph.procedures)
        latent_values[node] = [prior_distribution.sample()]
        if isinstance(prior_distribution, torch.distributions.Normal):
            loc = torch.clone(prior_distribution.loc)
            scale = torch.clone(prior_distribution.scale)
            distributions.append(Normal(loc,scale))
        elif isinstance(prior_distribution, torch.distributions.Gamma):
            concentration = torch.clone(prior_distribution.concentration)
            rate = torch.clone(prior_distribution.rate)
            distributions.append(Gamma(concentration, rate))
        elif isinstance(prior_distribution, torch.distributions.Exponential):
            rate = torch.clone(prior_distribution.rate)
            distributions.append(Exponential(rate))
        elif isinstance(prior_distribution, torch.distributions.Beta):
            concentration0 = torch.clone(prior_distribution.concentration0)
            concentration1 = torch.clone(prior_distribution.concentration1)
            distributions.append(Beta(concentration1, concentration0))
        elif isinstance(prior_distribution, torch.distributions.Dirichlet):
            concentration = torch.clone(prior_distribution.concentration)
            distributions.append(Dirichlet(concentration))
        elif isinstance(prior_distribution, torch.distributions.Bernoulli):
            logits = torch.clone(prior_distribution.logits)
            distributions.append(Bernoulli(logits=logits))
        elif isinstance(prior_distribution, torch.distributions.Categorical):
            probs = torch.clone(prior_distribution.probs)
            distributions.append(Categorical(probs=probs))
        else:
            raise ValueError(f'Unknown distribution type {prior_distribution} found.')
    return DistributionCollection(distributions)


def get_variational_distribution(program):
    if program == 1:
        mean, var = torch.tensor(1.), torch.sqrt(torch.tensor(5.))
        return Normal(mean, var)
    elif program == 2:
        mean = torch.zeros(2)
        var = torch.tensor([10., 10.])
        return Normal(mean, var)
    elif program == 3:
        raise NotImplementedError()
    elif program == 4:
        raise NotImplementedError()
    elif program == 5:
        m_dist = Normal(torch.tensor(0.), torch.tensor(5.))
        s_dist = Gamma(torch.tensor(9.), torch.tensor(2.))
        return DistributionCollection([m_dist, s_dist])


def get_valid_samples(graph, samples):
    """Check samples againts the prior support, return a mask of which samples are valid"""
    n_samples = samples[0].shape[0]
    valid_mask = torch.ones(n_samples).bool()
    for i in range(n_samples):
        latent_values = {}
        for latent, latent_samples in zip(graph.latents, samples):
            try:
                prior_distribution, _ = eval_expression(graph.link_functions[latent][0], [], latent_values, graph.procedures)
            except:
                valid_mask[i] = False
                break

            in_support = prior_distribution.support.check(latent_samples[i])
            if not in_support.all():
                valid_mask[i] = False
                break
            else:
                latent_values[latent] = [latent_samples[i]]
    return valid_mask

def compute_prior_probability(graph, samples):
    """For a given graphical model, compute p(x).

    Args:
        graph: A Graph object representing the graphical model.
        samples: A n_latents length list where each element is a set of samples for that latent.

    Returns:
        The probability of the samples based on the prior defined in the graph.
    """
    n_samples = samples[0].shape[0]
    prior_log_probs = torch.zeros(n_samples)
    for i in range(n_samples):
        latent_values = {}
        for latent_index, node in enumerate(graph.latents):
            latent_sample = samples[latent_index][i]
            prior_distribution, _ = eval_expression(graph.link_functions[node][0], [], latent_values, graph.procedures)
            prior_log_prob = prior_distribution.log_prob(latent_sample)
            if prior_log_prob.dim () > prior_log_probs[i].dim():
                prior_log_prob = prior_log_prob[0]
            prior_log_probs += prior_log_prob
            latent_values[node] = [latent_sample]
    return prior_log_probs


def compute_likelihood(graph, samples):
    """For a given graphical model, compute p(y|x) for the samples x.

    Args:
        graph: A Graph object representing the graphical model.
        samples: A n_latents length list where each element is a set of samples for that latent..

    Returns:
        The likelihood of the observations for each of the samples under the graphical model provided.
    """
    n_samples = samples[0].shape[0]
    log_likelihoods = torch.zeros(n_samples)
    for i in range(n_samples):
        latents = {node: [samples[latent_idx][i]] for latent_idx, node in enumerate(graph.latents)}
        for node in graph.observed:
            e1, e2 = graph.link_functions[node].sub_expressions
            d1, _ = eval_expression(e1, [], latents, graph.procedures)
            c2, _ = eval_expression(e2, [], latents, graph.procedures)
            log_likelihood = d1.log_prob(c2)
            if log_likelihood.dim() != log_likelihoods[i].dim():
                log_likelihood = log_likelihood[0]
            log_likelihoods[i] += log_likelihood
    return log_likelihoods


def run_BBVI(graph, mode, num_samples, tmax, wandb_name, verbose, program, n_steps=50000, samples_per_step=1):
    if mode != 'graph':
        raise ValueError('BBVI is only implemented for graph mode.')

    if program != 5:
        Q = get_variational_distribution_from_prior(graph)
    else:
        Q = get_variational_distribution(program)
    optimizer = Adam(Q.optim_params(), lr=0.01)

    for step in range(n_steps):

        samples = Q.sample([samples_per_step])
        samples_valid = get_valid_samples(graph, samples)
        if samples_valid.sum() == 0:
            continue
        samples = [latent_samples[samples_valid] for latent_samples in samples]

        log_Q = Q.log_prob(samples)
        if log_Q.dim() > 1:
            log_Q = log_Q.sum(dim=-1)

        if not isinstance(samples, list):
            samples = [samples]
        prior_log_prob = compute_prior_probability(graph, samples)
        log_likelihood = compute_likelihood(graph, samples)
        log_P = prior_log_prob + log_likelihood

        log_W = log_P - log_Q

        ELBO_loss = -(log_Q * (log_W.detach())).mean()
        ELBO_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Q: {Q.params()}, ELBO: {ELBO_loss}')

    final_latent_samples = Q.sample([num_samples])
    r_samples = []
    for i in range(num_samples):
        latent_values = {}
        for latent, latent_samples in zip(graph.latents, final_latent_samples):
            latent_values[latent] = [latent_samples[i]]
        r_sample, _ = eval_expression(graph.return_expression, [], latent_values, graph.procedures)
        r_samples.append(r_sample)
    return torch.stack(r_samples)




