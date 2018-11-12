# Design Documents


```eval_rst
.. toctree::
   :glob:
   :maxdepth: 1

   *
```

## Overview
If you want to propose making a major change to the codebase rather than a simple feature addition, it's helpful to fill out and send around a design proposal document **before you go through all the work of implementing it**. This allows the community to better evaluate the idea, highight any potential downsides, or propose alternative solutions ahead of time and save unneeded effort.

For smaller feature requests just file an issue and fill out the feature request template.

### What is considered a "major change" that needs a design proposal?

Any of the following should be considered a major change:
* Anything that changes a public facing API such as model definition or inference.
* Any other major new feature, subsystem, or piece of functionality

Example issues that might need design proposals include [#75](https://github.com/amzn/MXFusion/issues/75),
[#40](https://github.com/amzn/MXFusion/issues/40),
[#24](https://github.com/amzn/MXFusion/issues/24), or
[#23](https://github.com/amzn/MXFusion/issues/23).

### Process to submit a design proposal
Fill out the template below, add it to this folder on your fork, and make a pull request against the main repo. If it is helpful, feel free to include some proof of concept or mockup code to demonstrate the idea more fully. The point isn't to have fully functioning or tested code of your idea, but to help communicate the idea and how it might look with the rest of the community.

## Template

The basic template should include the following things:

### Motivation
Describe the problem to be solved.

### Public Interfaces
Describe how this changes the public interfaces of the library (if at all). Also describe any backwards compatibility strategies here if this will break an existing API.

### Proposed Changes
Describe the new thing you want to do. This may be fairly extensive and have large subsections of its own. Or it may be a few sentences, depending on the scope of the change.

### Rejected Alternatives
What are the other alternatives you considered and why are they worse? The goal of this section is to help people understand why this is the best solution now, and also to prevent churn in the future when old alternatives are reconsidered.

## Acknowledgements

This process is heavily inspired and taken from the [Kafka Improvement Processes](https://cwiki.apache.org/confluence/display/KAFKA/Kafka+Improvement+Proposals).
