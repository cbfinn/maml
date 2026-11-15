""" Code for second derivatives not implemented in TensorFlow library. """
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
    # MaxPoolGrad(orig_input, orig_output, grad) is piecewise constant in
    # orig_input and orig_output, so their gradients are zero. Its gradient
    # w.r.t. grad needs a MaxPoolGradGrad op, which TensorFlow < 1.2 lacks, so it
    # is also set to zero here (no second-order term through max pooling).
    # TensorFlow >= 1.2 registers the exact gradient and maml.py skips this file.
    gradgrad0 = array_ops.zeros(shape = array_ops.shape(op.inputs[0]), dtype=op.inputs[0].dtype)
    gradgrad1 = array_ops.zeros(shape = array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype)
    gradgrad2 = array_ops.zeros(shape = array_ops.shape(op.inputs[2]), dtype=op.inputs[2].dtype)
    return (gradgrad0, gradgrad1, gradgrad2)
