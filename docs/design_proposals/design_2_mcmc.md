# Possible design extensions to support MCMC sampling algorithms

Eric Meissner(2018-11-12)

## Motivation

To support the first MCMC (Markov-Chain Monte Carlo) inference algorithm in MXFusion, a few issues need to be resolved. These are primarily due to the Markovian part of MCMC, requiring a one-step time series style approach to generate proposal samples from the previous samples. This document discusses designs for those issues that are extensible to future MCMC algorithms, but don't necessarily solve the complete set of time-series modelling issues. I will take the Metropolis-Hastings algorithm as standard in this document, as that's the first MCMC inference we'll be implementing and it is more general(and more difficult to implement in MXFusion) than say Gibbs sampling or Hamiltonian Monte Carlo.

### Problem Setup
The two primary design choices to be made are:
1. Where are MCMC samples stored / transferred to future inference algorithms? What is the interface for this, and how are they initialized?
2. How do we take most recent sample in the chain, ```z_t```, and use it to generate the next sample ```z_t+1```.
 * This is non-trivial in MXFusion because it builds a circular reference into the proposal's FactorGraph. ```z_t+1``` depends on ```z_t``` but for a Model (without time-series support) saying a Variable depends on itself, ```m.z = Distribution(inputs=[m.z, ...]```, doesn't work right now.


## Proposed Changes


For problem (1), I propose we extend the InferenceParameters class (either as a subclass or a flag during initialization controlled by the Inference method) to include storing parameters for latent variables (LV), which is what the output samples of an MCMC algorithm are. Throughout the Inference method, these LV parameters would keep an up-to-date list of samples generated. In this solution, MCMC samples are easily serialized with no extra effort, and the existing TransferInference class can be extended readily to support reuse of these samples in later algorithms.
* Downside: This doesn't out-of-the-box allow for things like multi-chain generation and storage without storing the parameters elsewhere outside of the Inference loop, but that could easily be added in the future as needed. Another downside is that it complicates the InferenceParameters a bit, but either introducing subclasses or a flag that changes the behavior to include LV generation, and InferenceParameters

For problem (2), I propose:
* Developing the InferenceAlgorithm.compute() method as a method that takes in variables (LVs ```theta_t``` and parameters) for timestep ```t``` and outputs the samples for the LVs for the next timestep in the chain ```t+1``` (```theta_t+1```).
* The Inference class handles initializing parameters and LVs for ```t=0```, calling the InferenceAlgorithm.compute() in a loop for the number of samples requested, and managing the correct timestep of LV sample values that are passed into InferenceAlgorithm.compute().
* Treat the proposal distribution as a Posterior FactorGraph, leveraging the standard draw_samples and log_pdf methods.
* Introducing mapping variables into this proposal distribution that cut the connections needed to go from ```theta_t``` to ```theta_t+1```.
 * The transfer of values one timestep to the next for LVs (i.e. to generate a sample at timestep ```3```, it needs as input the sample from timestep ```2``` which was the output at that timestep) is handled in the Inference class.

 A working proof of concept is attached in the original pull request for this. It does not have a correctly extended InferenceParameters class, nor does it store the sample values in InferenceParameters directly. It successfully trained the Getting Started tutorial and the PPCA tutorial after adding a prior to ```m.w``` and using ```variance=1e-4``` for the proposal distributions.


## Rejected Alternatives

An alternative solution to problem (1) is to not store the MCMC samples at all in the Inference object, and simply return the samples after the Inference method completes. The main downsides to this are that the user then has to maintain those samples for use in future algorithms, and manually serialize those samples. I also don't see an easy alternative solution to the problem of initializing/storing the latent variables correctly **during Inference** without replicated code from InferenceParameters needing to go into the MCMCInference class.

An alternative solution to problem (2) is ...
