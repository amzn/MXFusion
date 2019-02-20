# Serialization

## Saving

Saving your work in MXFusion is straightforward.
Saving an inference method will save the model, any additional graphs used in the inference method, the state of the parameters at the end of the inference method, and any relevant configuration and constants used for the inference. Simply call ```.save``` on the run inference method you want to save.

```python
infr.save(prefix=PREFIX)
```


The model and other graphs are all saved into a single JSON file using NetworkX's [JSON graph format](https://networkx.github.io/documentation/latest/reference/readwrite/json_graph.html). MXFusion ModelComponents are serialized into JSON objects (see ```mxfusion.util.graph_serialization```) and Modules are stored recursively as sub-graphs inside the same JSON structure. The most important information attached to a ModelComponent when it is saved is its place in the graph topology, its UUID, and its 'name' attribute (the model class attribute name used to refer this model component), as these are how we reload the graph and parameters in successfully later. It is important to note that only a skeleton of the graphs are actually saved and that the model creation code must be re-run at load time.


> *If you're curious why we can't save all of it, this is because we can't save things with arbitrary code like MXNet Block's that users can use within MXFusion as Functions in their FactorGraphs.)*


The parameters of and constants are saved using MXNet Gluon's serialization functionality, as a map of UUID to numerical value. Any other inference configurations are saved into a simple JSON file.

## Loading back to MXFusion

Loading back in inference results in MXFusion is also straightforward. Before loading, re-run the model/posterior and inference creation code that you ran when you trained the Inference method. Then call ```.load``` on the newly created inference method, passing in the relevant file names from the save step.

```python
infr2.load(graphs_file=PREFIX+'_graphs.json',
           parameters_file=PREFIX+'_params.json')
```

Internally, the loading process has 3 major steps. The first is to reload the graphs and parameters from files into memory. The second is to reconcile those loaded graphs and parameters with the current model and inference method. The third is to load the rest of the configuration.

The first step uses NetworkX to load back in the graphs, which it loads into skeleton FactorGraphs (not full Models or Posteriors, and only basic ModelComponents with connections not Variables and Factors with information like what type of distribution the Factor is) because only minimal topology and naming information is saved during serialization. It uses MXNet to load the parameters back into Gluon Parameters.

The second step traverses the loaded skeleton FactorGraphs and attempts to match the variables in those graphs to the corresponding variables in the current model that you ran before loading the model . When it finds a match, it loads the corresponding parameter into the current inference's parameters and makes a note of this match. It then performs this process recursively for all variables in all of the graphs. We use the UUIDs and names of the variables, and the topology of the graphs as relevant information during the reconciliation process but it's not perfect and may fail sometimes due to ambiguities in the graph. If this happens, try naming more variables explicitly in the graph by attaching them to the graph directly, i.e. ```m.node = Variable()``` (or filing an issue!)

The third step simply loads from the JSON configuration file into the inference method and relevant configuration.

# Hybridize and loading from native MXNet

TBD See issue \#109.
