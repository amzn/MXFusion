# Inference

Notes about inference in MXFusion.

## Inference Algorithms

MXFusion currently supports stochastic variational inference.

### Variational Inference

Variational inference is an approximate inference method that can serve as the inference method over generic models built in MXFusion. The main idea of variational inference is to approximate the (often intractable) posterior distribution of our model with a simpler parametric approximation, referred to as a variational posterior distribution. The goal is then to optimize the parameters of this variational posterior distribution to best approximate our true posterior distribution. This is typically done by minimizing the lower bound of the logarithm of the marginal distribution:

\begin{equation}
\log p(y|z) = \log \int_x p(y|x) p(x|z) \geq \int_x q(x|y,z) \log \frac{p(y|x) p(x|z)}{q(x|y,z)} = \mathcal{L}(y,z), \label{eqn:lower_bound_1}
\end{equation}

where $(y|x) p(x|z)$ forms a probabilistic model with $x$ as a latent variable, $q(x|y)$ is the variational posterior distribution, and the lower bound is denoted as $\mathcal{L}(y,z)$. By then taking a natural exponentiation of $\mathcal{L}(y,z)$, we get a lower bound of the marginal probability denoted as $\tilde{p}(y|z) = e^{\mathcal{L}(y,z)}$.

A technical challenge with VI is that the integral of the lower bound of a probabilistic module with respect to external latent variables may not always be tractable.
Stochastic variational inference (SVI) offers an approximated solution to this new intractability by applying Monte Carlo Integration. Monte Carlo Integration is applicable to generic probabilistic distributions and lower bounds as long as we are able to draw samples from the variational posterior.

In this case, the lower bound is approximated as
\begin{equation}
\mathcal{L}(l, z) \approx \frac{1}{N} \sum_i \log \frac{p(l|y_i)e^{\mathcal{L}(y_i,z)}}{q(y_i|z)}, \quad \mathcal{L}(y_i, z) \approx \frac{1}{M} \sum_j \log \frac{p(y_i|x_j) p(x_j|z)}{q(x_j|y_i, z)} ,
\end{equation}
where $y_i|z \sim q(y|z)$, $x_j|y_i,z \sim q(x|y_i,z)$ and $N$ is the number of samples of $y$ and $M$ is the number of samples of $x$ given $y$. Note that if there is a closed form solution of $\tilde{p}(y_i|z)$, the calculation of $\mathcal{L}(y_i,z)$ can be replaced with the closed-form solution.

Let's look at a simple model and then see how we apply stochastic variational inference to it in practice using MXFusion.

###  Creating a Posterior
 TODO


## Examples
* [PPCA](../../examples/notebooks/ppca_tutorial.ipynb)

## Saving Inference Results
