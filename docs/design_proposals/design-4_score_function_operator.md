# A new operator to allow transparent usage of score function variational inference

Zhenwen Dai (2019-06-07)

## Motivation

A significant portion of stochastic variational inference (SVI) methods are developed based on the score function approach. Different from SVI with reparameterizable distributions, when applying a score function method, the gradient estimators for the variational posterior and the model parameters follow two separate formulation. Let $p(y, x | \theta)$ be the prior distribution of a probabilistic model, where $y$ is the observed variable, $x$ is the latent variable and $\theta$ is the parameters of the model. Let $q_{\phi}(x)$ be the variational posterior of $x$. With Monte Carlo sampling, the variationa lower bound is approximated by
$$
\mathcal{L} \approx \frac{1}{N}\sum_{i} \log \frac{p(y, x_i|\theta)}{q_{\phi}(x_i)}, \quad x_i \sim q_{\phi}(x),
$$
where $N$ is the number of drawn samples for $x$. For a non-reparameterizable distribution, the gradient with respect to $\phi$ cannot be obtained through samples $x_i$. When applying a score function method, we need to separate the gradient estimator for $\theta$ and $\phi$, i.e.,
$$
\nabla_{\theta} \mathcal{L} \approx \frac{1}{N}\sum_{i}  \nabla_{\theta} \log p(y, x_i|\theta),
$$
$$
\nabla_{\phi} \mathcal{L} \approx \frac{1}{N}\sum_{i} \log p(y, x_i|\theta) \nabla_{\phi} \log  q_{\phi}(x_i).
$$
The gradient estimators for $\theta$ and $\phi$ are very different. To facilitate auto-differentiation for both $\theta$ and $\phi$ with a single objective function, we can derive the following training objective by using the stop gradient operator,
$$
\mathcal{O} = \frac{1}{N}\sum_{i} \log p(y, x_i|\theta) + \mathcal{S}\left(\log p(y, x_i|\theta)\right) \log  q_{\phi}(x_i)
$$
where $\mathcal{S}(\cdot)$ is the stop gradient operator, which stops the auto-differentiation from going into the input of the operator. This is the currently implementation of the score function inference algorithm, but the limitation is that this objective cannot be consumed by any downstream calculation because the value does not correspond to the correct expectation as shown in the first equation.   

To allow the score function method to be used transparently in PPL, we need the outcome of the score function method to be $\mathcal{L}$ and, when performing auto-differentiation, the gradient should be estimated from $\mathcal{O}$.


## Proposed Changes

To address the above challenge, we propose a new operator that enable the separation of the forward and backward behavior in auto-differentiation. If we abstract the pattern of the forward and backward calculation in the above challenge, we get a function $l(p, q)$ depends on two variables $p$ and $q$. The forward calculation of the function is
$$
l(p, q) = p,
$$
and the backward calculation is
$$
\frac{\partial o}{\partial p} = \frac{\partial o}{\partial l}, \quad \frac{\partial o}{\partial q} = \frac{\partial o}{\partial l} p,
$$
where $o$ is the scalar final objective of the whole auto-differentiation.

We can implement the above operator as a MXNet customer operator such as
```python
def score_func_grad(p, q):
    ...
```

The score function variational inference can be implemented as
```python
import mxnet as mx

x_samples = q_x.draw_samples()
log_p = model.log_pdf(x_samples, y)
log_q = q_x.log_pdf(x_samples)
bound = mx.nd.mean(score_func_grad(log_p, log_q) - mx.nd.BlockGrad(log_q))
```

## Rejected Alternatives

I haven't seen an alternative solution yet.

## Reference

Cox, D. R.; Hinkley, D. V. (1974). Theoretical Statistics. Chapman & Hall. ISBN 0-412-12420-3.
