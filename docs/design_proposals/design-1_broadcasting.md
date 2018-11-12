# Changing the design of variable broadcasting for factors

Zhenwen Dai (2018-10-26)

## Motivation

The current design of the interface for creating a probabilistic distribution in MXFusion supports automatic broadcasting if the shapes of one or all the input variables are smaller than the expected shape. For example, we can create a random variable of the size (3, 4) with the shape of its mean being (1,) and the shape of its variance being (4,) as follows:
```py
m = Model()
m.mean = Variable(shape=(1,))
m.variance = Variable(shape=(4,))
m.x = Normal.define_variable(mean=m.mean, variance=m.variance, shape=(3, 4))
```
Although it is a handy feature, it causes confusions for some users because they are not familiar with the common broadcasting rule.

It also leads a big challenge when implementing new distributions. The current design requires users to write one function decorator for each of the two main APIs: log_pdf and draw_samples for any multi-variate distributions. Such a function decorator does two things:
1. If an input variable has a shape that is smaller than the expected one (computed from the shape of the random variable), it broadcasts the shape of the input variable to the expected shape.
2. For any the variables, the number of samples is not one. If the numbers of samples of more than one variables are more than one, the number of samples of these variables are assumed to be the same. It broadcast the size of the first dimension to the number of samples.

This mechanism is complicated and makes it hard get contributions.


## Proposed Changes

To simplify the logic of broadcasting, a natural choice is to let users take care of the first part of the broadcasting job. We still need to take care of the second part of the broadcasting as it is invisible from users. After the changes, the example with the new interface should be like
```py
m = Model()
m.mean = Variable(shape=(1,))
m.variance = Variable(shape=(4,))
m.x = Normal.define_variable(mean=broadcast_to(m.mean, (3, 4)),
                             variance=broadcast_to(m.variance, (3, 4)), shape=(3, 4))
```

In the above example, the operator ```broadcast_to``` takes care of broadcasting a variable to another shape and the ```Normal``` instance expects that all the inputs have the same shape as the random variable, which is (3, 4) in this case. This simplifies the implementation of distributions.

For broadcasting the number of samples, the mechanism for distributions and function evaluations can be unified. A boolean class attribute ```broadcastable``` will be added to ```Factor```. This attribute controls whether to expose the extra dimension of samples into internal computation logic. With this attribute being ```False```, a developer can implement the computation without the extra sample dimension, which is much straight-forward.

With this simplification, the function decorator is not necessary anymore. The class structure of a distribution will look like
```py
class Distribution(Factor):

  def log_pdf(self, F, variables, targets=None):
    # Take care of broadcasting of the number of samples.
    # or looping through the number of samples
    # call _log_pdf_implementation

  def _log_pdf_implementation(self, F, **kwargs):
    # The inherited classes will implement the real computation here.
```

## Rejected Alternatives

A possible solution to reduce the complexity of the broadcasting implementation without changing the interface is to add a shape inference function to each distribution. The shape inference will return the expected shapes of individual input variables given the shape of the random variable. Then, the ```broadcast_to``` operator just needs to be called inside the ```log_pdf``` function to hide the step of shape broadcasting.

The drawbacks of this approach is
- The shape inference needs to be implemented for each multi-variate distribution.
- The broadcasting logic is not consistent with function evaluations and modules, which causes confusions to users.
