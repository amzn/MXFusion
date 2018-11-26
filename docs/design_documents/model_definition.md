# Model Definition

MXFusion is a library for doing probabilistic modelling.

Probabilistic Models can be
categorized into directed graphical models (DGM, Bayes Net) and undirected
graphical models (UGM). Most popular probabilistic models
are DGMs, so MXFusion currently only supports DGMs.

A DGM can be fully defined using 3 basic components: deterministic functions,
probabilistic distributions, and random variables. As such, those are the primary ModelComponents in MXFusion.

## Model
The primary data structure of MXFusion is the [FactorGraph](https://en.wikipedia.org/wiki/Factor_graph). FactorGraphs contain Variables and Factors. The FactorGraph exposes methods that Inference algorithms call such as drawing samples from the FactorGraph or computing the log pdf of a set of Variables contained in the FactorGraph.

When you want to start modelling, construct a Model object and start attaching ModelComponents to it.
You can then see the Model's components by

```Python
m = Model()
m.v = Variable()
print(m.components)
```

When a ModelComponent is attached to a Model, it is automatically updated in the Model's internal data structures and will be included in any subsequent inference operations over the model.

## Model Components

All ModelComponents in MXFusion are identified uniquely by a UUID.

### Variables
In a model, there are typically four types of variables: a random variable
following a probabilistic distribution, a variable which is the outcome of a
deterministic function, a parameter (with no prior distribution), and a
constant. The definitions of first two types of variables will be discussed
later. The latter two types of variables can be defined with the following
statement:

```Python
m.v = Variable(shape=(10,), constraint=PositiveTransformation())
```

At this stage, you do not need to specify whether *v* is a parameter or constant,
because, if it is a constant, its value will be provided during
inference, otherwise it will be treated as a parameter.

A typical example of when a constant would be specified at inference time is the size (shape) of an
observed variable, which is known when data is provided. In the above
example, we specify the name of the variable, the shape of the variable and
the constraint that the variable has. It defines a 10-dimension vector whose
values are always positive (v>=0).

### Factors

#### Distributions
In a probabilistic model, random variables relate to each other through
probabilistic distributions.

During model definition, the typical interface to generate a 2 dimensional
random variable ```m.x``` from a zero mean unit variance Gaussian distribution
looks like:

```python
from mxnet.ndarray import array

m.x = Normal.define_variable(mean=array([0, 0]), variance=array([1, 1]), shape=(2,))
```

The two dimensions are
independent to each other and both follow the same Gaussian
distribution. The parameters or shape of a distribution can also be variables, for
example:

```python
from mxnet.ndarray import array

m.mean = Variable(shape=(2,))
m.y_shape = Variable()
m.y = Normal.define_variable(mean=m.mean, variance=array([1, 1]), shape=(m.y_shape,))
```

MXFusion also allows users to specify a prior distribution over pre-existing
variables. This is particularly handy for interfacing with neural networks in
MXNet because it allows you to set priors over parameters in an existing Gluon
Block, such as a neural network implementation. The API for specifying a prior
distribution looks like:

```Python
m.x = Variable(shape=(2,))
m.x.set_prior(Gaussian(mean=array([0, 0]), variance=array([1, 1]))
```

The above code defines a variable ```m.x``` and sets the prior distribution of
each dimension of ```m.x``` to be a scalar unit Gaussian distribution.

In many cases, we apply the same prior distribution to multiple dimensions. In the above example, we simply want to set the individual dimensions of ```m.x``` to follow a zero-mean and unit-variance Gaussian. A more elegant way to define the above prior distribution is to make use of the broadcasting rule of multi-dimensional arrays:
```Python
from mxfusion.components.functions.operators import broadcast_to

m.x.set_prior(Gaussian(mean=broadcast_to(array([0]), m.x.shape),
                       variance=broadcast_to(array([1]), m.x.shape)))
```
Note that the shape of ```m.x``` may not always be available. In those cases, it is better to explicitly define the shape to be broadcasted to.

Because Models are FactorGraphs, it is common to want to know what ModelComponents come before or after a particular component in the graph. These are accessed through the ModelComponent properties ```successors``` and ```predecessors```.

```python
m.mean = Variable()
m.var = Variable()
m.y = Normal.define_variable(mean=m.mean, variance=m.var)
```


#### Functions
The last building block of probabilistic models are deterministic functions. The
ability to define sophisticated functions allows users to build expressive
models with a family of standard probabilistic distributions. As MXNet already
provides full functionality for defining a function and automatically
evaluating its gradients, Functions in MXFusion are a wrapper over the
functions in MXNet's Gluon interface. Functions are defined in standard MXNet
syntax and provided to the MXFusion Function wrapper as below.

First we define a function in MXNet Gluon syntax using a Block object:

```Python
class F(mx.gluon.Block):    
    def forward(self, x, y):
        return x*2+y
```

Then we create an MXFusion Function instance by passing in our Gluon function
instance:

```Python
f_gluon = F()
m.f_mf = MXFusionGluonFunction(f_gluon)
```

Then this MXFusion Function can be called using MXFusion variables and its
outcome will another variable[s] representing the outcome of the function:

```python
m.x = Variable(shape=(2,))
m.y = Variable(shape=(2,))
m.f = f_mf(x, y)
```


## FAQ
* Why don't you support undirected graphical models (UGM)?
 * A UGM is typically defined in terms of a set of potential functions.
 Each potential function is a non-negative function that is defined on a subset of variables in a model.
 The joint probability distribution of an UGM is defined as the product of all the potential functions
 divided by a normalization term (known as a partition function).
 * The notation of a DGM and an UGM can be unified into a factor graph,
 where a factor can be either a probabilistic distribution or a potential function.
  **In our implementation, the distribution UI is inherited from the factor abstract class,
   which enables future extension to support UGM** , although inference algorithms
    for UGM will be completely different.
