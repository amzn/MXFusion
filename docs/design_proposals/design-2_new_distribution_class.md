# Introduce a new class of execution time distribution

Zhenwen Dai (2019-04-16)

## Motivation

The current type of distribution class in MXFusion is a "symbolic" notation for probabilistic model and variational posterior definition. When developing sophisticated inference algorithm, one often faces the need of creating some intermediate distribution objects that do not exist in model/posterior definition. In the implementation of current inference algorithms, those distributions are explicitly represented in terms of their parameters. This is not ideal for memory organization and code reusing. If there are distribution objects for execution, those intermediate distributions can be better organization in an object-oriented fashion. Many distribution computation functions can be implemented on top of those execution time distribution classes such as Kullback-Leiber divergence. A good example of such distribution classes is the distribution class in Tensorflow (tf.distrbution).

## Proposed Changes

Implement a family of distribution classes that are used at execution time. This type distribution class differs from the distribution in model definition in the following ways:
1. An instance of such a distribution may not exist in model or posterior definition.
2. The attributes of such a distribution are MXNet arrays.
3. Many computational helper functions can be built around this family of distributions.

With this family of distributions, we will shift all the computational code in the current distribution classes (the log_pdf and sample_samples functions) to the family of execution time distributions.

A normal distribution of the execution time class will look like:
```python
class Normal:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def log_pdf(self, random_variable):
        logvar = np.log(2 * np.pi) / -2 + mx.nd.log(self.variance) / -2
        logL = mx.nd.broadcast_add(logvar, mx.nd.broadcast_div(F.square(
            mx.nd.broadcast_minus(random_variable, self.mean)), -2 * variance))
        return logL

    def draw_samples(self, num_samples=1):
        out_shape = (num_samples,) + self.mean.shape
        return mx.nd.broadcast_add(mx.nd.broadcast_mul(self._rand_gen.sample_normal(
            shape=out_shape),
            mx.nd.sqrt(self.variance)), self.mean)

    def kl_divergence(self, other_dist):
        ...
```

## Rejected Alternatives
