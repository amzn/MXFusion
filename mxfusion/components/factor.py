"""Factor module.

.. autosummary::
    :toctree: _autosummary

"""

from copy import copy
from .model_component import ModelComponent


class Factor(ModelComponent):
    """
    A factor represents a relation among multiple variables in a model such as a distribution, a function or a module. It consists of a list of output
    variables and optionally a list of input variables.

    The ``inputs`` and ``outputs`` argument of ``__init__`` holds the input and output of the factor, which are represented in Python dict. The key of a variable in
    the dict is the name of the variable referred in the context of the factor, e.g., the mean and variance of a normal distribution. The value of a
    variable is the reference to the variable in memory. Both input and output variables are accessible as class attributes.

    The ``inputs`` and ``outputs`` argument of ``__init__`` can be:

    * A list of variables
    * An empty list (no input/output variables)
    * None (the input/output variables are not provided yet.)

    Note that the ``outputs`` argument should never be an empty list, as a factor always outputs some variables.

    :param inputs: the input variables of the factor.
    :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
    :param outputs: the output variables of the factor.
    :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
    """

    def __getattr__(self, value):
        if value.startswith("__"):
            """
            When python copies objects, it begins by checking for ``__setstate__()`` which doesn't exist, so it calls ``__getattr__()``. Our implementation then
            calls the ``self.inputs`` getter before the object is fully prepared because ``__init__()`` never gets called during the copy. This causes an infinite
            recursion to ``__getattr__()``. By skipping magic methods with "__" prefix, we allow the object to initialize correctly during copying.

            # TODO this is very inefficient, can be improved.
            """
            raise AttributeError(value)

        if value in self._input_names:
            for name, node in self.inputs:
                if name == value:
                    return node

        if value in self._output_names:
            for name, node in self.outputs:
                if name == value:
                    return node
        else:
            raise AttributeError("''%s' object has no attribute '%s'" % (type(self), value))

    def __init__(self, inputs, outputs, input_names, output_names):
        super(Factor, self).__init__()
        self._check_name_conflict(inputs, outputs)
        self._input_names = input_names if input_names is not None else []
        self._output_names = output_names if output_names is not None else []
        self.predecessors = inputs if inputs is not None else []
        self.successors = outputs if outputs is not None else []


    def __repr__(self):
        out_str = str(self.__class__.__name__)
        if self.predecessors is not None:
            out_str = out_str + \
                '(' + ', '.join([str(name) + '=' + str(var) for
                                 name, var in self.predecessors]) + ')'
        return out_str

    def replicate_self(self, attribute_map=None):
        """
        This functions is a copy constructor for the object.
        In order to perform copy construction we first call ``__new__()`` on the class which creates a blank object.
        We then initialize that object using the method's standard init procedures, and do any extra copying of attributes.

        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        :param outputs: new output variables of the factor.
        :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        """

        replicant = self.__class__.__new__(self.__class__)

        Factor.__init__(
            replicant, None, None, copy(self.input_names),
            copy(self.output_names))
        replicant._uuid = self.uuid
        return replicant

    def _check_name_conflict(self, inputs, outputs):
        if inputs is not None and outputs is not None:
            intersect = set([v for _, v in inputs]) & set([v for _, v in outputs])
            if intersect:
                raise RuntimeError("The inputs and outputs variables of " +
                                   self.__class__.__name__ + " have name conflict: " + str(intersect) + ".")

    @property
    def inputs(self):
        """
        Return a list of nodes whose edges point into this node.
        """
        if self.graph is not None:
            pred = {e['name']: v for v, e in self.graph.pred[self].items()}
            return [(name, pred[name]) for name in self.input_names]
        else:
            return self._predecessors

    @property
    def outputs(self):
        """
        Return a list of nodes pointed to by the edges of this node.
        """
        if self.graph is not None:
            succ = {e['name']: v for v, e in self.graph.succ[self].items()}
            return [(name, succ[name]) for name in self.output_names]
        else:
            return self._successors

    @inputs.setter
    def inputs(self, inputs):
        """
        Set Input variables of the factor.

        :param inputs: Input variables of the factor.
        :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)]
        """
        self.predecessors = inputs

    @outputs.setter
    def outputs(self, outputs):
        """
        Set Output variables of the factor.

        :param outputs: Output variables of the factor.
        :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)]
        """
        self.successors = outputs

    def set_outputs(self, variables):
        """
        TODO We don't actually support multi-output.
        """
        variables = [variables] if not isinstance(variables, (list, tuple)) else variables
        self.successors = [(name, variable) for name, variable in zip(self.output_names, variables)]

    def set_single_input(self, key, value):
        """
        Set a single input variable of a factor.

        :param key: the name of the input variable in the factor
        :type key: str
        :param value: the variable to be set
        :type value: Variable
        """
        inputs = [(k, value) if k == key else (k, v) for k, v in self.inputs]
        self.predecessors = inputs

    @property
    def input_names(self):
        """
        Return Input names.
        """
        return self._input_names

    @property
    def output_names(self):
        """
        Return Output names.
        """
        return self._output_names

    def fetch_runtime_inputs(self, params):
        """
        The helper function to fetch the input variables from a set of
        variables according to the UUIDs of the input variables. It returns a
        dictionary of variables at runtime, where the keys are the name of the
        input variables and the values are the MXNet array at runtime. The
        returned dict can be directly passed into runtime functions of factors
        such as eval for functions and log_pdf and draw_samples for
        distributions.

        :param params: the set of variables where the input variables are
        fetched from.
        :type params: {str (UUID): MXNet NDArray or MXNet Symbol}
        :return: a dict of the input variables, where the keys are the name
        of the input variables and the values are the MXNet array at runtime.
        :rtype: {str (kernel name): MXNet NDArray or MXNet Symbol}
        """
        return {n: params[v.uuid] for n, v in self.inputs}

    def fetch_runtime_outputs(self, params):
        """
        The helper function to fetch the output variables from a set of
        variables according to the UUIDs of the output variables. It returns a
        dictionary of variables at runtime, where the keys are the name of the
        output variables and the values are the MXNet array at runtime. The
        returned dict can be directly passed into runtime functions of factors
        such as eval for functions and log_pdf and draw_samples for
        distributions.

        :param params: the set of variables where the output variables are
        fetched from.
        :type params: {str (UUID): MXNet NDArray or MXNet Symbol}
        :return: a dict of the output variables, where the keys are the name
        of the output variables and the values are the MXNet array at runtime.
        :rtype: {str (kernel name): MXNet NDArray or MXNet Symbol}
        """
        return {n: params[v.uuid] for n, v in self.outputs}
