""" Code for the MAML algorithm and network definitions. """
import numpy as np
#import special_grads
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize, sigmoid_xent, safe_get, init_conv_weights_xavier, init_bias, init_weights

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5, dim_state_input=None):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_state_input = dim_state_input
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        self.inner_loss_func = None
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [FLAGS.fc_hidden]*FLAGS.num_fc
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif 'push' in FLAGS.datasource:
            # loss func and model
            def mod_loss(x, y, **kwargs):
                return mse(x, y, multiplier=50, **kwargs)
            self.loss_func = mod_loss #lambda x, y, **kwargs: mse(x, y, multiplier=50.0)
            self.forward = self.forward_fp
            self.construct_weights = self.construct_fp_weights
            self.channels = 3

        elif 'omniglot' in FLAGS.datasource or 'rainbow' in FLAGS.datasource or FLAGS.datasource in ['miniimagenet','mnist']:
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
            if FLAGS.datasource == 'miniimagenet' or 'rainbow' in FLAGS.datasource:
                self.channels = 3
            elif 'siamese' in FLAGS.datasource:
                self.channels = 2   # 2 images
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
            if FLAGS.baseline == 'contextual':
                self.channels *= (FLAGS.update_batch_size + 1)
        else:
            raise ValueError('Unrecognized data source.')
        if FLAGS.learned_loss:
            self.inner_loss_func = self.learned_loss
        elif self.inner_loss_func is None:
            self.inner_loss_func = self.loss_func
        if FLAGS.context_var:
            if FLAGS.conv and self.classification:
                #context_channels = 3 # TODO - hardcoded
                #self.context_size = [self.dim_input]
                self.context_size = [10]
                #self.channels += 10
            else:
                self.context_size = [10]
                self.dim_input += self.context_size[0]

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.statea = tf.placeholder(tf.float32)
            self.stateb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        # selfs are placeholders, nonselfs will be overwritten
        inputa, inputb, labela, labelb, statea, stateb = self.inputa, self.inputb, self.labela, self.labelb, self.statea, self.stateb

        if FLAGS.test_on_train:
            inputa = inputb
            labela = labelb
            statea = stateb
        if FLAGS.baseline == 'contextual':
            # TODO - why is this not working?
            assert not FLAGS.test_on_train
            assert not FLAGS.context_var
            # dim 0 is task, dim 1 is examples, dim 2,3,4 is image
            # NOTE - this assumes that num classes = 1 (true for rainbow mnist, not for others)
            labela = labelb
            if FLAGS.conv and self.classification:
                pre_channels = int(self.channels / (FLAGS.update_batch_size + 1))
                inputa = tf.reshape(inputa, [-1, FLAGS.update_batch_size, self.img_size, self.img_size, pre_channels])
                inputb = tf.reshape(inputb, [-1, FLAGS.update_batch_size, self.img_size, self.img_size, pre_channels])
                # put inputb batch into channels
                new_inputa = tf.transpose(inputa, [0,2,3,1,4])
                new_inputa = tf.reshape(new_inputa, [-1, 1, self.img_size, self.img_size, FLAGS.update_batch_size*pre_channels])
            #else:
            #    new_inputa = tf.reshape(inputa, [-1, 1, self.dim_input*FLAGS.update_batch_size]
            inputa = tf.concat([tf.concat([tf.zeros([-1, 1, self.img_size, self.img_size, FLAGS.update_batch_size * pre_channels]), inputb[:,i:i+1]], -1) for i in range(FLAGS.update_batch_size)], 1)
            if FLAGS.conv and self.classification:
                inputa = tf.reshape(inputa, [-1, FLAGS.update_batch_size, self.img_size*self.img_size*self.channels])
            # Also make inputb the right shape to prevent issues
            inputb = inputa

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
                loss_weights = self.loss_weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                if FLAGS.pred_task and FLAGS.learn_loss:
                    self.loss_weights = self.construct_loss_weights()
                else:
                    self.loss_weights = None
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
                if 'push' in FLAGS.datasource:
                    task_inputa, task_inputb, task_labela, task_labelb, task_statea, task_stateb = inp
                else:
                    task_inputa, task_inputb, task_labela, task_labelb = inp
                    task_stateb = None
                    task_stateas = [None]*num_updates
                task_outputbs, task_lossesb = [], []
                if self.classification:
                    task_accuraciesb = []

                if FLAGS.inner_sgd:
                    c = FLAGS.update_batch_size * FLAGS.num_classes
                    task_inputas = [task_inputa[c*i:c*(i+1), :] for i in range(num_updates)]
                    task_labelas = [task_labela[c*i:c*(i+1), :] for i in range(num_updates)] 
                    if 'push' in FLAGS.datasource:
                        task_stateas = [task_statea[c*i:c*(i+1), :] for i in range(num_updates)]
                else:
                    task_inputas = [task_inputa]*num_updates
                    task_labelas = [task_labela]*num_updates
                    if 'push' in FLAGS.datasource:
                        task_stateas = [task_statea]*num_updates


                task_outputa, _ = self.forward(task_inputas[0], weights, reuse=reuse, ind=0, state_input=task_stateas[0])  # only reuse on the first iter
                task_lossa = self.inner_loss_func(task_outputa, task_labelas[0], sine_x=task_inputas[0], postupdate=False, loss_weights=self.loss_weights)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                if 'push' in FLAGS.datasource:
                    for key in gradients.keys():
                        gradients[key] = tf.clip_by_value(gradients[key], -10, 10)
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))

                if FLAGS.update_bn_only:
                    for key in weights.keys():
                        if 'scale' not in key and 'offset' not in key:
                            fast_weights[key] = weights[key]

                output, _ = self.forward(task_inputb, fast_weights, reuse=True, state_input=task_stateb)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, task_labelb, postupdate=True))

                for j in range(num_updates - 1):
                    output_j, _ = self.forward(task_inputas[j+1], fast_weights, reuse=True, ind=j+1, state_input=task_stateas[j+1])
                    loss = self.inner_loss_func(output_j, task_labelas[j+1], sine_x=task_inputa, postupdate=False, loss_weights=self.loss_weights)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    if 'push' in FLAGS.datasource:
                        for key in gradients.keys():
                            gradients[key] = tf.clip_by_value(gradients[key], -10, 10)
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))

                    if FLAGS.update_bn_only:
                        for key in weights.keys():
                            if 'scale' not in key and 'offset' not in key:
                                fast_weights[key] = weights[key]

                    output, _ = self.forward(task_inputb, fast_weights, reuse=True, state_input=task_stateb)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, task_labelb, postupdate=True))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    if 'siamese' not in FLAGS.datasource:
                        task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(task_labelas[0], 1))
                        for j in range(num_updates):
                            task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(task_labelb, 1)))
                    else:
                        correct_pred = tf.equal(tf.cast(task_outputa > 0, tf.int32), tf.cast(task_labelas[0], tf.int32))
                        task_accuracya = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                        for j in range(num_updates):
                            correct_pred = tf.equal(tf.cast(task_outputbs[j] > 0, tf.int32), tf.cast(task_labelb, tf.int32))
                            task_accuraciesb.append( tf.reduce_mean(tf.cast(correct_pred, tf.float32)) )
                    task_output.extend([task_accuracya, task_accuraciesb])

                #task_output.append(output_viz)

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                if 'push' in FLAGS.datasource:
                    unused = task_metalearn((inputa[0], inputb[0], labela[0], labelb[0], statea[0], stateb[0]), False)
                else:
                    unused = task_metalearn((inputa[0], inputb[0], labela[0], labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            #out_dtype.append(tf.float32)
            if 'push' in FLAGS.datasource:
                result = tf.map_fn(task_metalearn, elems=(inputa, inputb, labela, labelb, statea, stateb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            else:
                result = tf.map_fn(task_metalearn, elems=(inputa, inputb, labela, labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb = result

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
                var_list = None
                if FLAGS.learn_loss_only:
                    assert FLAGS.learned_loss
                    var_list = [var for var in tf.trainable_variables() if 'loss' in var.name]
                if FLAGS.zerok_shot:
                    if FLAGS.inner_sgd:
                        meta_objective = sum(self.total_losses2[:FLAGS.num_updates]) #+ self.total_loss1
                    else:
                        meta_objective = self.total_losses2[FLAGS.num_updates-1] + self.total_loss1
                else:
                    meta_objective = self.total_losses2[FLAGS.num_updates-1]
                self.gvs = gvs = optimizer.compute_gradients(meta_objective, var_list=var_list)
                if FLAGS.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
                tf.summary.scalar(prefix+'Optimized loss', meta_objective)
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
            if FLAGS.metatrain_iterations > 0:
                tf.summary.scalar(prefix+'Optimized loss', self.total_losses2[FLAGS.num_updates-1])
            else:
                tf.summary.scalar(prefix+'Optimized loss', self.total_loss1)

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    def learned_loss(self, pred, label=None, **kwargs):
        # Label is unused, to match the signature of other losses.
        #pred = tf.nn.softmax(pred)
        if FLAGS.label_in_loss:
            pred = tf.concat([pred,label], -1)
            #pred = tf.nn.softmax(pred) - label
        fc_init =  tf.contrib.layers.xavier_initializer(dtype=tf.float32)   
        if 'loss_weights' not in dir(self) or self.loss_weights is None:
            hidden_dim = 40
            self.loss_weights = {}
            if FLAGS.label_in_loss:
                self.loss_weights['w1'] = tf.Variable(fc_init([self.dim_output, hidden_dim]), name='loss_w1')  
            else:
                self.loss_weights['w1'] = tf.Variable(fc_init([self.dim_output, hidden_dim]), name='loss_w1') 
            #self.loss_weights['w2'] = tf.Variable(fc_init([hidden_dim, 1]), name='loss_w2')  
            self.loss_weights['b1'] = tf.Variable(tf.zeros([40]), name='loss_b1') 
            #self.loss_weights['b2'] = tf.Variable(tf.zeros([1]), name='loss_b2') 
        #hidden = tf.nn.relu(tf.matmul(pred, self.loss_weights['w1']) + self.loss_weights['b1'])  
        #loss = tf.square(tf.matmul(hidden, self.loss_weights['w2']) + self.loss_weights['b2'])  
        # don't square this?  
        #loss = tf.square(tf.matmul(pred, self.loss_weights['w1']) + self.loss_weights['b1'])  
        # logit mse
        loss = tf.reduce_sum(tf.square(pred))
        return loss   

    def construct_loss_weights(self):
        dtype = tf.float32
        fc_init =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        loss_weights = {}
        if FLAGS.label_in_loss:
            loss_weights['lw1'] = tf.Variable(fc_init([7, self.dim_hidden[0]]))
        else:
            loss_weights['lw1'] = tf.Variable(fc_init([6, self.dim_hidden[0]]))
        loss_weights['lb1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        loss_weights['lw2'] = tf.Variable(fc_init([self.dim_hidden[0], self.dim_hidden[1]]))
        loss_weights['lb2'] = tf.Variable(tf.zeros([self.dim_hidden[1]]))
        loss_weights['lw3'] = tf.Variable(fc_init([self.dim_hidden[1], 1]))
        loss_weights['lb3'] = tf.Variable(tf.zeros([1]))
        return loss_weights

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        dtype = tf.float32
        fc_init =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        weights = {}
        #weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['w1'] = tf.Variable(fc_init([self.dim_input, self.dim_hidden[0]]))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            if i <= FLAGS.fc_linear:
                #weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01/FLAGS.fc_linear))
                weights['w'+str(i+1)] = tf.Variable(fc_init([self.dim_hidden[i-1], self.dim_hidden[i]]))
            else:
                #weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
                weights['w'+str(i+1)] = tf.Variable(fc_init([self.dim_hidden[i-1], self.dim_hidden[i]]))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        #weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(fc_init([self.dim_hidden[-1], self.dim_output]))
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
            if i <= FLAGS.fc_linear:
                hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.identity, reuse=reuse, scope=str(i+1))
            else:
                hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

    def construct_fp_weights(self):
        weights = {}
        self.num_filters = 16
        self.n_conv_layers = 4
        self.strides = [[1, 2, 2, 1]]*self.n_conv_layers
        self.filter_sizes = [5]*self.n_conv_layers
        self.img_size = 125
        downsample_factor = 1
        self.conv_out_size = int(self.num_filters*2)

        # conv weights
        fan_in = self.channels
        for i in range(self.n_conv_layers):
            weights['wc%d' % (i+1)] = init_conv_weights_xavier([self.filter_sizes[i], self.filter_sizes[i], fan_in, self.num_filters], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
            weights['bc%d' % (i+1)] = init_bias([self.num_filters], name='bc%d' % (i+1))
            fan_in = self.num_filters

        # fc weights
        in_shape = self.conv_out_size
        # if you have state
        in_shape += self.dim_state_input
        self.bt_dim = 20
        in_shape += self.bt_dim
        self.conv_out_size_final = in_shape
        weights['context'] = safe_get('context', initializer=tf.zeros([self.bt_dim], dtype=tf.float32))

        self.n_layers = 3
        dim_hidden = [200]*(self.n_layers-1)
        dim_hidden.append(self.dim_output)
        for i in range(self.n_layers):
            weights['w_%d' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d' % i)
            weights['b_%d' % i] = init_bias([dim_hidden[i]], name='b_%d' % i)
            in_shape = dim_hidden[i]

        return weights

    def forward_fp(self, inp, weights, reuse=False, scope='', ind=None, state_input=None):
        # hacky way to tile the context variable without knowing the number of examples
        flatten_image = tf.reshape(inp, [-1, self.img_size*self.img_size*self.channels])
        context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), list(range(20))))
        context += weights['context']

        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])


        conv_layer = inp 
        for i in range(self.n_conv_layers):
            conv_layer = conv_block(conv_layer, weights['wc%d' % (i+1)], weights['bc%d' % (i+1)], reuse=reuse, scope=str(i))
        _, num_rows, num_cols, num_fp = conv_layer.get_shape()
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]

        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)
        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols
        x_map = tf.convert_to_tensor(x_map)
        y_map = tf.convert_to_tensor(y_map)
        x_map = tf.reshape(x_map, [num_rows * num_cols])
        y_map = tf.reshape(y_map, [num_rows * num_cols])
        features = tf.reshape(tf.transpose(conv_layer, [0,3,1,2]),
                                  [-1, num_rows*num_cols])
        softmax = tf.nn.softmax(features)
        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
        conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
        fc_input = tf.add(conv_out_flat, 0)
        fc_input = tf.concat(axis=1, values=[fc_input, context])
        fc_output = fc_input
        # for state input
        fc_output = tf.concat(axis=1, values=[fc_output, state_input])
        for i in range(self.n_layers):
            fc_output = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
            if i != self.n_layers - 1:
                fc_output = tf.nn.relu(fc_output)
        return fc_output, None



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

    def forward_conv(self, inp, weights, reuse=False, scope='', ind=None, **kwargs):
        # reuse is for the normalization parameters.
        if FLAGS.datasource == 'miniimagenet' or 'rainbow' in FLAGS.datasource:
            channels = 3 # TODO - don't hardcode.
        elif 'siamese' in FLAGS.datasource:
            channels = 2
        else:
            channels = 1
        if FLAGS.baseline == 'contextual':
            channels *= (FLAGS.update_batch_size + 1)
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

        # inputs are returned for visualization purposes
        return tf.matmul(hidden4, weights['w5']) + weights['b5'], None


