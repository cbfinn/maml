"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.



    To run rainbow MNIST comparison for Aravind, run:
        python3 main.py --datasource=contrainbow_mnist20 --cont_incl_cur=False  --metatrain_iterations=1000000 --num_updates=5 --update_lr=0.1 --update_batch_size=10 --num_classes=1 --logdir=logs/aravind_seed2 --num_filters=32 --aravind=True --cont_seed=2
    To run rainbow MNIST our method, run:
        python3 main.py --datasource=contrainbow_mnist20 --cont_incl_cur=True  --metatrain_iterations=1000000 --num_updates=5 --update_lr=0.1 --update_batch_size=10 --num_classes=1 --logdir=logs/ftml_seed2 --num_filters=32 --cont_seed=2
    To run rainbow MNIST from scratch, change to TASK_ITER/BATCH_ITER to 4000/100 in main and run:
        python3 main.py --datasource=contrainbow_mnist20  --metatrain_iterations=1000000 --num_updates=1 --update_lr=0.1 --update_batch_size=10 --num_classes=1 --logdir=logs/indep4k_seed2 --num_filters=32  --cont_seed=2 --train_only_on_cur=True --cont_finetune_on_all=False --baseline=oracle
    To run rainbow MNIST TOE, run:
    python3 main.py --datasource=contrainbow_mnist20  --metatrain_iterations=1000000 --num_updates=1 --update_lr=0.1 --update_batch_size=10 --num_classes=1 --logdir=logs/toe2k_seed2 --num_filters=32  --cont_seed=2 --cont_finetune_on_all=False --baseline=oracle
"""
import csv
import glob
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet or siamese_omniglot or cont_push')
flags.DEFINE_string('datadistr', 'stationary', 'stationary or continual1')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
# contextual means to concatenate the meta-training set as input (channel-wise concat if images)
flags.DEFINE_string('baseline', None, 'oracle, online, incl_task, contextual, or None')
flags.DEFINE_bool('context_var', False, 'whether or not to include context variable to append to input')
flags.DEFINE_bool('aravind', False, 'whether to be running comparison for Aravind')
flags.DEFINE_integer('cont_seed', 1, 'seed for task order.')

flags.DEFINE_bool('pred_task', False, 'whether or not to predict the task context rather than the label for inner update')
flags.DEFINE_bool('learn_loss', False, 'whether or not to use a learned loss (only supported with pred_task and sinusoid)')
flags.DEFINE_bool('label_in_loss', False, 'whether or not to include the label as input to the  learned loss (only supported with learn_loss and sinusoid)')
flags.DEFINE_bool('semisup_loss', False, 'whether or not to use a semi-supervised learned loss (only supported with pred_task and sinusoid)')

flags.DEFINE_bool('update_bn', False, 'whether or not to update the batch normalization variables in the inner update')
flags.DEFINE_bool('update_bn_only', False, 'whether or not to *only* update the batch normalization variables in the inner update')
flags.DEFINE_bool('shuffle_tasks', False, 'whether or not to shuffle the tasks, when loading data, only for rainbow mnist.')

flags.DEFINE_bool('nearest_neighbor', False, 'test time only - eval classification using nearest neighbor in pixel space')

flags.DEFINE_bool('alternate_grad_meta', False, 'whether or not to alternate between plain GD steps and meta-GD steps')

flags.DEFINE_bool('learned_loss', False, 'whether or not to use a learned inner loss function')
flags.DEFINE_bool('test_on_train', False, 'inner and outer dataset the same')
flags.DEFINE_bool('inner_sgd', False, 'whether or not to SGD in the inner loop')
flags.DEFINE_float('pixel_dropout', 0.0, 'pixel dropout percentage')
flags.DEFINE_bool('learn_loss_only', False, 'outer objective only includes loss function parameters')

## Continual learning flags
flags.DEFINE_bool('zerok_shot', False, 'meta objective of both zero shot and k shot.')
flags.DEFINE_bool('cont_finetune_on_all', True, 'finetune on all data so far.')
flags.DEFINE_bool('cont_incl_cur', True, 'include the current task in meta-training.')
flags.DEFINE_bool('train_only_on_cur', False, 'only train on the current task.')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_integer('fc_hidden', 40, 'number of hidden units for fc nets.')
flags.DEFINE_integer('num_fc', 2, 'number of fc hidden layers for fc nets.')
flags.DEFINE_integer('fc_linear', 0, 'number of fc hidden layers that should be linear for fc nets.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

flags.DEFINE_bool('incl_switch', False, 'if True, switch top and bottom of MNIST digits with 50% probability')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 20
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.logdir + '/val' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    sess.graph.finalize()

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []
    inputa, inputb, labela, labelb = None, None, None, None

    train_examples = FLAGS.num_classes * FLAGS.update_batch_size
    if FLAGS.inner_sgd:
        train_examples *= FLAGS.num_updates

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        # TODO - modify this for pushing?
        # **NORMAL is % 1k, % 25
        # **SLOW is % 2k, % 50
        # **EXTRASLOW is % 4k, % 100
        TASK_ITER = 2000
        BATCH_ITER = 50
        # initialize continual learning at this iteration
        INIT_CONT = TASK_ITER / 2
        if itr >= INIT_CONT and 'cont' in FLAGS.datasource:  # used to be itr > 1000 and itr % 100
            if itr >= TASK_ITER/2 and itr % TASK_ITER == 0:
                data_generator.add_task()
                #tf.global_variables_initializer().run()
            elif itr % BATCH_ITER == 0:
                data_generator.add_batch()
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate(itr=itr)

            if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'online' or FLAGS.baseline=='incl_task':
                if 'sinusoid' in FLAGS.datasource: # siamese already has task id encoded in the input
                  batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                  for i in range(FLAGS.meta_batch_size):
                      batch_x[i, :, 1] = amp[i]
                      batch_x[i, :, 2] = phase[i]
            if FLAGS.pred_task:
                if 'sinusoid' in FLAGS.datasource: # siamese already has task id encoded in the input
                  batch_y = np.concatenate([batch_y, np.zeros([batch_y.shape[0], batch_y.shape[1], 2])], 2)
                  for i in range(FLAGS.meta_batch_size):
                      batch_y[i, :, 1] = amp[i]
                      batch_y[i, :, 2] = phase[i]
            if FLAGS.baseline == 'online' and itr % 2 == 1:
                batch_x, batch_y, amp, phase = last_batch
            elif FLAGS.baseline == 'online':
                last_batch = batch_x, batch_y, amp, phase

            inputb = batch_x[:, train_examples:, :] # b used for testing
            labelb = batch_y[:, train_examples:, :]
            inputa = batch_x[:, :train_examples, :]
            labela = batch_y[:, :train_examples, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
            if 'push' in FLAGS.datasource:
                feed_dict[model.statea] = amp[:, :train_examples, :] # b used for testing
                feed_dict[model.stateb] = amp[:, train_examples:, :]

        if itr < FLAGS.pretrain_iterations or (FLAGS.baseline == 'online' and itr % 2 == 1) or (FLAGS.alternate_grad_meta and itr % 2 == 1):
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        #if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
        if (itr!=0) and itr % SUMMARY_INTERVAL == 0:
            if 'generate' not in dir(data_generator):
                assert not FLAGS.inner_sgd # not yet supported
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                    #input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False, itr=itr)

                if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'online' or FLAGS.baseline=='incl_task':
                    if 'sinusoid' in FLAGS.datasource: # siamese already has task id encoded in the input
                      batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                      for i in range(FLAGS.meta_batch_size):
                          batch_x[i, :, 1] = amp[i]
                          batch_x[i, :, 2] = phase[i]
                if FLAGS.pred_task:
                    if 'sinusoid' in FLAGS.datasource: # siamese already has task id encoded in the input
                      batch_y = np.concatenate([batch_y, np.zeros([batch_y.shape[0], batch_y.shape[1], 2])], 2)
                      for i in range(FLAGS.meta_batch_size):
                          batch_y[i, :, 1] = amp[i]
                          batch_y[i, :, 2] = phase[i]

                # using -train_examples for val.
                # finetune_on_all means include all data thus far
                if not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
                    train_examples = -train_examples
                val_inputa = batch_x[:, :train_examples, :]
                val_inputb = batch_x[:, train_examples:, :]
                val_labela = batch_y[:, :train_examples, :]
                val_labelb = batch_y[:, train_examples:, :]
                feed_dict = {model.inputa: val_inputa, model.inputb: val_inputb,  model.labela: val_labela, model.labelb: val_labelb, model.meta_lr: 0.0}
                if 'push' in FLAGS.datasource:
                    feed_dict[model.statea] = amp[:, :train_examples, :] # b used for testing
                    feed_dict[model.stateb] = amp[:, train_examples:, :]
                if not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
                    train_examples = -train_examples
                input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]
                if model.classification:
                    input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

            input_tensors.append(model.summ_op)
            result = sess.run(input_tensors, feed_dict)
            val_writer.add_summary(result[-1], itr)
            if itr % TEST_PRINT_INTERVAL == 0:
                print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    train_examples = num_classes * FLAGS.update_batch_size
    if FLAGS.inner_sgd:
        train_examples *= max(test_num_updates, FLAGS.num_updates)

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'online' or FLAGS.baseline == 'incl_task':
                if FLAGS.datasource == 'sinusoid':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    batch_x[0, :, 1] = amp[0]
                    batch_x[0, :, 2] = phase[0]
            if FLAGS.pred_task:
                if 'sinusoid' in FLAGS.datasource: # siamese already has task id encoded in the input
                  batch_y = np.concatenate([batch_y, np.zeros([batch_y.shape[0], batch_y.shape[1], 2])], 2)
                  for i in range(FLAGS.meta_batch_size):
                      batch_y[0, :, 1] = amp[0]
                      batch_y[0, :, 2] = phase[0]

            inputa = batch_x[:, :train_examples, :]
            inputb = batch_x[:,train_examples:, :]
            labela = batch_y[:, :train_examples, :]
            labelb = batch_y[:,train_examples:, :]

            # nearest neighbor code:
            num_right = 0
            num_total = 0
            if FLAGS.nearest_neighbor:
                for i in range(num_classes):
                    test_img = inputb[:,i,:]
                    test_label = labelb[0,i]
                    diff = np.mean( (test_img - inputa)**2, axis=2)
                    idx = np.argmin(diff)
                    train_label = labela[0,idx]
                    if np.all(train_label == test_label):
                        num_right += 1
                    num_total += 1
                metaval_accuracies.append(float(num_right) / num_total)
            else:
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if not FLAGS.nearest_neighbor:
            if model.classification:
                if 'generate' not in dir(data_generator):
                    result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
                else:
                    result = sess.run([model.total_accuracy1] + model.total_accuracies2, feed_dict)
            else:  # this is for sinusoid
                result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
            metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'w') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 20  # normally 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            if FLAGS.train == True:
                if FLAGS.baseline == 'oracle':
                    test_num_updates = 1
                else:
                    test_num_updates = 10
            else:
                test_num_updates = 20 # 50

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        assert not FLAGS.inner_sgd # not yet supported
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    elif  'rainbow_mnist' in FLAGS.datasource:
        #assert FLAGS.update_batch_size < 5  # only 5 images per task
        # was 5, now udpate_batch_size*2
        if FLAGS.inner_sgd:
            test_num_updates = FLAGS.num_updates
            data_generator = DataGenerator(FLAGS.update_batch_size*(FLAGS.num_updates+1), FLAGS.meta_batch_size)
        else:
            data_generator = DataGenerator(min(FLAGS.update_batch_size*2, 100), FLAGS.meta_batch_size)
    else:
        assert not FLAGS.inner_sgd # not yet supported
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)


    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'online' or FLAGS.baseline == 'incl_task':
        assert FLAGS.datasource == 'sinusoid' or 'siamese' in FLAGS.datasource or 'mnist' in FLAGS.datasource
        if FLAGS.datasource == 'sinusoid':
            dim_input = 3
        else:
            dim_input = data_generator.dim_input
    else:
        dim_input = data_generator.dim_input
    dim_state_input = data_generator.dim_state_input
    if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'contextual':
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    if FLAGS.pred_task and FLAGS.datasource == 'sinusoid':
        dim_output = 3  # 2 for pre-update, 1 for post-update. Will just contain all 3 for now

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot': # not including siamese omniglot
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates, dim_state_input=dim_state_input)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr) +'.metalr' + str(FLAGS.meta_lr)

    if FLAGS.datadistr != 'stationary':
        exp_string += FLAGS.datadistr +'.'
    if FLAGS.learned_loss:
        if FLAGS.label_in_loss:
            exp_string += 'learned_labloss'
        else:
            exp_string += 'learned_loss'
    if FLAGS.shuffle_tasks:
        exp_string += 'task_shuffle'
    if FLAGS.test_on_train:  # TODO - implement test_on_train
        exp_string += 'test_on_train'
    if FLAGS.pixel_dropout > 0:
        exp_string += 'pdropout'
    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.fc_hidden != 40:
        exp_string += 'hidden' + str(FLAGS.fc_hidden)
    if FLAGS.num_fc != 2:
        exp_string += 'numfc' + str(FLAGS.num_fc)
    if FLAGS.fc_linear != 0:
        exp_string += 'fclin' + str(FLAGS.fc_linear)
    if FLAGS.pred_task:
        exp_string += '.predtask.'
    if FLAGS.learn_loss:
        if FLAGS.semisup_loss:
            exp_string += 'fixsemilearn2losswinput.'
            assert not FLAGS.label_in_loss
        else:
            if FLAGS.label_in_loss:
                exp_string += 'learn2losswinplbl.'
            else:
                exp_string += 'learn2losswinput.'
    exp_string += ''
    if FLAGS.incl_switch:
        exp_string += 'switch0.5'


    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.alternate_grad_meta:
        exp_string += '.alternate.'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.context_var:
        exp_string += 'context_fc'
    if FLAGS.update_bn or FLAGS.update_bn_only: # update bn vars in update
        exp_string += 'updatebn'
        if FLAGS.update_bn_only:
            exp_string += 'only'
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')
    #exp_string += 'conv13doublin'
    if FLAGS.inner_sgd:
        exp_string += 'sgd'
    if FLAGS.zerok_shot:
        exp_string += '0kshot'

    resume_itr = 0
    model_file = None

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    print('Done constructing graph. Initializating session and variables.')
    if FLAGS.baseline == 'oracle':
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if model_file is None and FLAGS.learn_loss_only:
            # Loading from first oracle
            #oracle_dir = glob.glob(FLAGS.logdir + '/cls' + '*oracle*')
            oracle_dir = glob.glob(FLAGS.logdir + '/cls' + '*learned_loss*')
            if oracle_dir:
                print('Optionally loading from other checkpoint')
                model_file = tf.train.latest_checkpoint(oracle_dir[0])
                loaded_oracle = True
                #loader = tf.train.Saver([var for var in tf.trainable_variables() if 'loss' not in var.name], max_to_keep=10)
                loader = tf.train.Saver([var for var in tf.trainable_variables()], max_to_keep=10)
                # if you don't want to load from oracle, set model_file = None
                import pdb; pdb.set_trace()
        else:
            loaded_oracle = False
            loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
            if model_file:
                ind1 = model_file.index('model')
                resume_itr = int(model_file[ind1+5:])
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            print("Restoring model weights from " + model_file)
            loader.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
