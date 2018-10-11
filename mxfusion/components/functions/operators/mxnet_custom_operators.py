import mxnet as mx

class GammaLn(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation for the log gamma function.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        import scipy as sp
        x = in_data[0].asnumpy()
        y = sp.special.gammaln(x)
        self.assign(out_data[0], req[0], mx.nd.array(y, dtype=in_data[0].dtype))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation for the log gamma function.

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        import scipy as sp
        y = out_data[0].asnumpy()
        x = in_data[0].asnumpy()
        dy = out_grad[0].asnumpy()
        dx = dy*sp.special.digamma(x)
        self.assign(in_grad[0], req[0], mx.nd.array(dx, dtype=out_grad[0].dtype))

@mx.operator.register("gammaln")
class GammaLnProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(GammaLnProp, self).__init__(True)

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return GammaLn()

def gammaln(alpha):
    # import pdb; pdb.set_trace()
    return mx.nd.Custom(alpha, op_type='gammaln')
