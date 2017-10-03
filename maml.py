""" Code for the MAML algorithm and network definitions. """
import numpy as np
#import special_grads
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize, sigmoid_xent

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [FLAGS.fc_hidden, FLAGS.fc_hidden]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif 'omniglot' in FLAGS.datasource or FLAGS.datasource in ['miniimagenet','mnist']:
            if 'siamese' in FLAGS.datasource:
                self.loss_func = sigmoid_xent
            else:
                self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward=self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
            elif 'siamese' in FLAGS.datasource:
                self.channels = 2   # 2 images
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
        else:
            raise ValueError('Unrecognized data source.')
        if FLAGS.context_var:
            if FLAGS.conv and self.classification:
                #context_channels = 3 # TODO - hardcoded
                #self.context_size = [self.dim_input]
                self.context_size = [10]
                #self.channels += 10
            else:
                self.context_size = [50]
                self.dim_input += self.context_size[0]

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                if FLAGS.context_var:
                    self.weights['context_var'] = tf.get_variable('context_var', [1]+self.context_size, initializer=tf.contrib.layers.xavier_initializer())
                    weights['context_var'] = self.weights['context_var']

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))

                if FLAGS.update_bn_only:
                    for key in weights.keys():
                        if 'scale' not in key and 'offset' not in key:
                            fast_weights[key] = weights[key]

                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))

                    if FLAGS.update_bn_only:
                        for key in weights.keys():
                            if 'scale' not in key and 'offset' not in key:
                                fast_weights[key] = weights[key]

                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    if 'siamese' not in FLAGS.datasource:
                        task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                        for j in range(num_updates):
                            task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    else:
                        correct_pred = tf.equal(tf.cast(task_outputa > 0, tf.int32), tf.cast(labela, tf.int32))
                        task_accuracya = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                        for j in range(num_updates):
                            correct_pred = tf.equal(tf.cast(task_outputbs[j] > 0, tf.int32), tf.cast(labelb, tf.int32))
                            task_accuraciesb.append( tf.reduce_mean(tf.cast(correct_pred, tf.float32)) )
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if FLAGS.baseline == 'online':
                self.pretrain_op = tf.train.AdamOptimizer(self.update_lr).minimize(total_loss1)
            else:
                self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                if FLAGS.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
                tf.summary.scalar(prefix+'Optimized loss', self.total_losses2[FLAGS.num_updates-1])
            else:
                tf.summary.scalar(prefix+'Optimized loss', self.total_loss1)
                #optimizer = tf.train.AdamOptimizer(self.meta_lr)
                #self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        if FLAGS.update_bn or FLAGS.update_bn_only:
            raise NotImplementedError('update bn not yet supported for fc net')
        if FLAGS.context_var:
            context = tf.tile(weights['context_var'], [FLAGS.update_batch_size, 1])
            inp = tf.concat([inp, context], 1)
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv1b'] = tf.get_variable('conv1b', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1b'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.update_bn or FLAGS.update_bn_only:
            weights['offset1'] = tf.Variable(tf.zeros((self.dim_hidden,)))
            weights['scale1'] = tf.Variable(tf.ones((self.dim_hidden,)))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.update_bn or FLAGS.update_bn_only:
            weights['offset2'] = tf.Variable(tf.zeros((self.dim_hidden,)))
            weights['scale2'] = tf.Variable(tf.ones((self.dim_hidden,)))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3b'] = tf.get_variable('conv3b', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3b'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.update_bn or FLAGS.update_bn_only:
            weights['offset3'] = tf.Variable(tf.zeros((self.dim_hidden,)))
            weights['scale3'] = tf.Variable(tf.ones((self.dim_hidden,)))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.update_bn or FLAGS.update_bn_only:
            weights['offset4'] = tf.Variable(tf.zeros((self.dim_hidden,)))
            weights['scale4'] = tf.Variable(tf.ones((self.dim_hidden,)))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            if FLAGS.context_var:
                weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5+10, self.dim_output], initializer=fc_initializer)
            else:
                weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        else:
            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        if FLAGS.datasource == 'miniimagenet':
            channels = 3 # TODO - don't hardcode.
        elif 'siamese' in FLAGS.datasource:
            channels = 2
        else:
            channels = 1
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        if FLAGS.context_var:
            num_examp = int(inp.get_shape()[0])

        """
        if FLAGS.context_var:
            #channels = self.channels - 3  #self.context_size[-1] # TODO - specific to RGB.
            context = weights['context_var']

            #context = tf.tile(context, [self.img_size, 1])
            #context = tf.expand_dims(context, 0)
            #context = tf.tile(context, [self.img_size, 1, 1])
            #context = tf.expand_dims(context, 0)
            #context = tf.tile(context, [num_examp, 1, 1, 1])

            #context = tf.expand_dims(tf.expand_dims(context, 0), 0)
            #context = tf.tile(context, [num_examp, self.img_size, self.img_size, 1])

            context = tf.tile(context, [num_examp, 1])

            context = tf.reshape(context, [num_examp, self.img_size,self.img_size,10])
        else:
        """
        channels = self.channels

        #if (FLAGS.baseline and 'context_var' == FLAGS.baseline) or FLAGS.context_var:
        #    inp = tf.concat([inp, context], 3)
        if FLAGS.update_bn or FLAGS.update_bn_only:
            scales = [weights['scale'+str(i)] for i in range(1,5)]
            offsets = [weights['offset'+str(i)] for i in range(1,5)]
            bn_vars = zip(scales, offsets)
        else:
            bn_vars = [None]*4

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0', bn_vars[0])

        hidden1 = conv_block(hidden1, weights['conv1b'], weights['b1b'], reuse, scope+'0b', bn_vars[0], False, activation=tf.identity)

        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1', bn_vars[1])
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2', bn_vars[2])

        hidden3 = conv_block(hidden3, weights['conv3b'], weights['b3b'], reuse, scope+'2b', bn_vars[0], False, activation=tf.identity)

        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3', bn_vars[3])
        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        if FLAGS.context_var:
            context = tf.tile(weights['context_var'], [num_examp, 1])
            hidden4 = tf.concat([hidden4, context], 1)

        return tf.matmul(hidden4, weights['w5']) + weights['b5']


