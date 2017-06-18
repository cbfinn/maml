""" Code for second derivatives not implemented in TensorFlow library. """
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
    gradient = gen_nn_ops._max_pool_grad(op.inputs[0], op.outputs[0],
            grad, op.get_attr("ksize"), op.get_attr("strides"),
            padding=op.get_attr("padding"), data_format=op.get_attr("data_format"))
    gradgrad1 = array_ops.zeros(shape = array_ops.shape(op.inputs[1]), dtype=gradient.dtype)
    gradgrad2 = array_ops.zeros(shape = array_ops.shape(op.inputs[2]), dtype=gradient.dtype)
    return (gradient, gradgrad1, gradgrad2)
