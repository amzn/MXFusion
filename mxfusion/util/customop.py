import mxnet as mx
from mxnet.operator import CustomOp, CustomOpProp
from .util import parse_string_to_tuple


class BroadcastToWithSamplesOp(CustomOp):
    def __init__(self, isSamples, shape, **kwargs):
        self.isSamples = isSamples
        self.shape = shape
        super(BroadcastToWithSamplesOp, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        a = in_data[0]
        n_dim = len(self.shape)
        if self.isSamples:
            t_shape = (a.shape[0],) + (1,) * (n_dim - len(a.shape)) + a.shape[1:]
        else:
            t_shape = (1,) * (n_dim - len(a.shape)) + a.shape

        a_reshape = mx.nd.reshape(a, shape=t_shape)
        out = mx.nd.broadcast_to(a_reshape, shape=self.shape)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        a_shape = in_data[0].shape
        if self.isSamples:
            grad = mx.nd.reshape(out_grad[0], shape=(a_shape[0], -1,) + a_shape[1:])
            a_grad = mx.nd.sum(grad, axis=1)
        else:
            grad = mx.nd.reshape(out_grad[0], shape=(-1,) + a_shape)
            a_grad = mx.nd.sum(grad, axis=0)
        self.assign(in_grad[0], req[0], a_grad)


@mx.operator.register("broadcast_to_w_samples")
class BroadcastToWithSamplesOpProp(CustomOpProp):
    def __init__(self, **kwargs):
        self.isSamples = kwargs['isSamples'].lower() == 'true'
        self.shape = parse_string_to_tuple(kwargs['shape'])
        super(BroadcastToWithSamplesOpProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['out']

    def infer_shape(self, in_shapes):
        return in_shapes, (self.shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes, **kwargs):
        return BroadcastToWithSamplesOp(isSamples=self.isSamples,
                                        shape=self.shape, **kwargs)


def broadcast_to_w_samples(F, data, shape, isSamples=True):
    if F is mx.nd:
        n_dim = len(shape)
        if isSamples:
            num_samples = max(data.shape[0], shape[0])
            t_shape = (data.shape[0],) + (1,) * (n_dim - len(data.shape)) + data.shape[1:]
            shape = (num_samples,) + shape[1:]
        else:
            t_shape = (1,) * (n_dim - len(data.shape)) + data.shape

        data_reshape = F.reshape(data, shape=t_shape)
        return F.broadcast_to(data_reshape, shape=shape)
    else:
        return F.Custom(data, op_type="broadcast_to_w_samples",
                        isSamples=isSamples, shape=shape)
