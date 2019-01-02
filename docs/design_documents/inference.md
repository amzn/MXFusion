# Inference

## Overview

Inference in MXFusion is broken down into a few logical pieces that can be combined together as necessary. MXFusion relies on MXNet's Gluon as the underlying computational engine.

The highest level object you'll deal with will be derived from the ```mxfusion.inference.Inference``` class. This is the outer loop that drives the inference algorithm, holds the relevant parameters and models for training, and handles serialization after training. At a minimum, ```Inference``` objects take as input the ```InferenceAlgorithm``` to run. On creation, an ```InferenceParameters``` object is created and attached to the ```Inference``` method which will store and manage (MXNet) parameters during inference.

Currently there are two main Inference subclasses: ```GradBasedInference``` and ```TransferInference```. An obvious third choice would be some kind of MCMC sampling Inference method.

The first primary class of Inference methods is ```GradBasedInference```, which is for those methods that involve a gradient-based optimization. We only support the gradient optimizers that are available in MXNet for now. When using gradient-based inference methods (```GradBasedInference```), the Inference class takes in a ```GradLoop``` in addition to the ```InferenceAlgorithm```. The ```GradLoop``` determines how the gradient computed in the ```InferenceAlgorithm``` is used to update model parameters. The two available implementations of ```GradLoop``` are ```BatchInferenceLoop``` and ```MinibatchInferenceLoop```, which correspond to gradient-based optimization in batch or mini-batch mode.

The second type of Inference method is ```TransferInference```. These are methods that take as an additional parameter the ```InferenceParameters``` object from a previous Inference method. An example of a ```TransferInference``` method is the ```VariationalPosteriorForwardSampling``` method, which takes as input a VariationalInference method that has already been trained and performs forward sampling through the variational posterior.

A basic example to run variational inference with a meanfield posterior over some model looks like the following. See the next section for mathematical details on variational inference.

### First Example

First we create the model.  The model creation function is dummy here, but this applies to almost any model. See the [Model Definiton](../model_definition.md) file for details on model creation. Then we define the observed variables in our model, and apply the convenience method for creating a factorized Gaussian posterior to that model, and get the posterior ```q```.

```py
m = make_model()
observed = [m.y, m.x]
q = create_Gaussian_meanfield(model=m, observed=observed)
```

Then we define what ```InferenceAlgorithm``` we want to run, and initialize it with the model, posterior, and observation pattern we defined above. This is used to initialize the ```GradBasedInference``` object, which creates a data structure to manage parameters of the model at this stage.

```py
alg = StochasticVariationalInference(model=m, observed=observed, posterior=q)
infr = GradBasedInference(inference_algorithm=alg)
```

Then, we run the Inference method, passing in the data as keyword arguments, matching the observation pattern we defined previously. This will create and initialize parameters for the variational posterior and any model parameters, and optimize the standard KL-divergence loss function to match the variational posterior to the model's posterior. We run it for 1000 iterations.

```
infr.run(max_iter=1000, y=y, x=x)

```

## Inference Algorithms

MXFusion currently supports stochastic variational inference. We provide a convenience method to generate a Gaussian meanfield posterior for your model, but the interface is flexible enough to allow defining a specialized posterior over your model as required. See the ```mxfusion.inference``` module of the documentation for a full list of supported inference methods.

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

Variational inference is based around the idea that you can approximate your true model's, possibly complex, posterior distribution with an approximate variational posterior that is easy to compute. A common choice of approximate posterior is the Gaussian meanfield, which factorizes each variable as being drawn from a Normal distribution independent from the rest.

This can be done easily for a given model by calling the ```mxfusion.inference.create_Gaussian_meanfield``` function and passing in your model.

You can also define more complex posterior distributions to perform inference over if you know something more about your problem. See the [../../examples/notebooks/ppca_tutorial.ipynb](PPCA tutorial) for a detailed example of this process.


## Saving and Loading Inference Results
 Saving and reloading inference results is managed at the ```Inference``` level in MXFusion. Once you have an ```Inference``` object that has been trained, you save the whole thing by running:

 ```py
  inference.save('my_inference_prefix')
 ```

 This will save down all relevent pieces of the inference algorithm to files beginning with the prefix passed in at save time. These files include: MXNet parameter files, json files containing the model's topology, and any Inference configuration such as the number of samples it was run with.

When reloading a saved inference method, you must re-run the code used to generate the original models and Inference method, and then load the saved parameters back into the new objects. An example is shown below:

In process 1:
```py

x = np.random.rand(1000, 1)
y = np.random.rand(1000, 1)

m = make_model()

observed = [m.y, m.x]
q = create_Gaussian_meanfield(model=m, observed=observed)
alg = StochasticVariationalInference(num_samples=3, model=m, observed=observed, posterior=q)
infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
infr.initialize(y=y, x=x)
infr.run(max_iter=1, learning_rate=1e-2, y=y, x=x)

infr.save(prefix=PREFIX)

```

At some future time, in another process:
```py
x = np.random.rand(1000, 1)
y = np.random.rand(1000, 1)

m2 = make_model()

observed2 = [m2.y, m2.x]
q2 = create_Gaussian_meanfield(model=m2, observed=observed2)
alg2 = StochasticVariationalInference(num_samples=3, model=m2, observed=observed2, posterior=q2)
infr2 = GradBasedInference(inference_algorithm=alg2, grad_loop=BatchInferenceLoop())
infr2.initialize(y=y, x=x)

# Load previous parameters
infr2.load(primary_model_file=PREFIX+'_graph_0.json',
           secondary_graph_files=[PREFIX+'_graph_1.json'],
           parameters_file=PREFIX+'_params.json',
           inference_configuration_file=PREFIX+'_configuration.json',
           mxnet_constants_file=PREFIX+'_mxnet_constants.json',
           variable_constants_file=PREFIX+'_variable_constants.json')


```

## Inference Internals

Inference in MXFusion happens in a few steps.

The first thing for a variational inference method is to create a ```Posterior``` from the ```Model```, which makes a copy of the model that can then be changed without altering the structure of the original model while allowing the user to logically reference the same variable in the model and posterior.

When the ```InferenceAlgorithm``` object is created, references to the ```Model``` and ```Posterior``` objects are kept but no additional MXNet memory or parameters are allocated at this time.

When the ```Inference``` object is created, again, references to the graph objects are kept and an ```InferenceParameters``` object is created, but no MXNet memory is allocated yet.

Some ```Inference``` classes need their ```initialize(...)``` methods be called before calling ```run(...)```, but most can be called by simply calling ```run(...)``` with the appropriate arguments, and it will call initialize before proceeding with the run step.

When ```run(**kwargs)``` is called, the 3 primary steps happen:
1. ```Inference.initialize()``` is called if not already initialized. This derives the correct shapes of everything from the data passed in via ```kwargs``` and initializes all of the MXNet Parameter objects needed for the computation.
2. ```Inference.create_executor()``` is called (which calls it's ```InferenceAlgorithm.create_executor()```'s method) to create an ObjectiveBlock. This is an MXNet Gluon HybridBlock object. This is the primary computational graph object which gets executed to perform inference in MXFusion.
 * If desired, this block can be hybridized and saved down into a symbolic graph for reloading by passing in ```hybridize=True``` when initializing your ```Inference``` object. See MXNet Gluon documentation on [hybrid mode](https://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html) for more details.
3. The ```ObjectiveBlock``` or ```executor``` created in the last step is now run, running data through the MXNet compute graph that was constructed.
