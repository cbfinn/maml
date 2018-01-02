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
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or or polynomial or omniglot or miniimagenet or siamese_omniglot')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, online, incl_task, or None')
flags.DEFINE_bool('context_var', False, 'whether or not to include context variable to append to input')
flags.DEFINE_bool('amp_only', False, 'only include amplitude in task description')
flags.DEFINE_bool('zerok_shot', False, 'meta objective of both zero shot and k shot.')

flags.DEFINE_bool('max_ent', False, 'whether or not to use max ent objective')

flags.DEFINE_bool('update_bn', False, 'whether or not to update the batch normalization variables in the inner update')
flags.DEFINE_bool('update_bn_only', False, 'whether or not to *only* update the batch normalization variables in the inner update')

flags.DEFINE_bool('l1_loss', False, 'whether or not to use l1 loss with sinusoid ')
flags.DEFINE_bool('learned_loss', False, 'whether or not to use an inner learned loss')

flags.DEFINE_bool('nearest_neighbor', False, 'test time only - eval classification using nearest neighbor in pixel space')

flags.DEFINE_bool('alternate_grad_meta', False, 'whether or not to alternate between plain GD steps and meta-GD steps')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_bool('inner_sgd', False, 'if True, run SGD in inner loop rather than batch GD.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_integer('fc_hidden', 40, 'number of hidden units for fc nets.')
flags.DEFINE_integer('num_fc', 2, 'number of fc hidden layers for fc nets.')
flags.DEFINE_integer('fc_linear', 0, 'number of fc hidden layers that should be linear for fc nets.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test_ensemble', False, 'True to test normally. False to test ensemble of models.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator, resume_itr=0, test_num_updates=None):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid' or FLAGS.datasource == 'polynomial':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    train_examples = num_classes * FLAGS.update_batch_size
    if FLAGS.inner_sgd:
        train_examples *= max(test_num_updates, FLAGS.num_updates)


    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            if 'polynomial' in FLAGS.datasource:
              batch_x, batch_y, d1, d2 ,d3, d4, d5, d6, d7, d8, d9, d10 = data_generator.generate()
            else:
              batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'online' or FLAGS.baseline=='incl_task':
                if 'sinusoid' in FLAGS.datasource: # siamese already has task id encoded in the input
                  if FLAGS.amp_only:  # only include amplitude
                      batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 1])], 2)
                      for i in range(FLAGS.meta_batch_size):
                          batch_x[i, :, 1] = amp[i]
                  else:
                      batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                      for i in range(FLAGS.meta_batch_size):
                          batch_x[i, :, 1] = amp[i]
                          batch_x[i, :, 2] = phase[i]
                elif 'polynomial' in FLAGS.datasource: # siamese already has task id encoded in the input
                  batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 10])], 2)
                  for i in range(FLAGS.meta_batch_size):
                      batch_x[i, :, 1] = d1[i]
                      batch_x[i, :, 2] = d2[i]
                      batch_x[i, :, 3] = d3[i]
                      batch_x[i, :, 4] = d4[i]
                      batch_x[i, :, 5] = d5[i]
                      batch_x[i, :, 6] = d6[i]
                      batch_x[i, :, 7] = d7[i]
                      batch_x[i, :, 8] = d8[i]
                      batch_x[i, :, 9] = d9[i]
                      batch_x[i, :, 10] = d10[i]
            if FLAGS.baseline == 'online' and itr % 2 == 1:
                batch_x, batch_y, amp, phase = last_batch
            elif FLAGS.baseline == 'online':
                last_batch = batch_x, batch_y, amp, phase

            inputa = batch_x[:, :train_examples, :]
            labela = batch_y[:, :train_examples, :]
            inputb = batch_x[:, train_examples:, :] # b used for testing
            labelb = batch_y[:, train_examples:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations or (FLAGS.baseline == 'online' and itr % 2 == 1) or (FLAGS.alternate_grad_meta and itr % 2 == 1):
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
        #import pdb; pdb.set_trace() # input_tensors.append(model.neg_entropy_estimate)

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
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid' and FLAGS.datasource != 'polynomial':
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                    #input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :train_examples, :]
                inputb = batch_x[:, train_examples:, :]
                labela = batch_y[:, :train_examples, :]
                labelb = batch_y[:, train_examples:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 1200 #1200

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    train_examples = num_classes * FLAGS.update_batch_size
    if FLAGS.inner_sgd:
        train_examples *= max(test_num_updates, FLAGS.num_updates)

    if FLAGS.test_ensemble:
        # plot ensemble variance vs. task MSE
        # in distribution points in one color, out of distribution in another
        model_file1 = 'logs/sine1/seed1/cls_5.mbs_25.ubs_20.numstep5.updatelr0.001.metalr0.001hidden100context_fcnonorm/model69999'
        model_file2 = 'logs/sine1/seed2/cls_5.mbs_25.ubs_20.numstep5.updatelr0.001.metalr0.001hidden100context_fcnonorm/model69999'
        model_file3 = 'logs/sine1/seed3/cls_5.mbs_25.ubs_20.numstep5.updatelr0.001.metalr0.001hidden100context_fcnonorm/model69999'
        model_file4 = 'logs/sine1/seed4/cls_5.mbs_25.ubs_20.numstep5.updatelr0.001.metalr0.001hidden100context_fcnonorm/model69999'
        model_file5 = 'logs/sine1/seed5/cls_5.mbs_25.ubs_20.numstep5.updatelr0.001.metalr0.001hidden100context_fcnonorm/model69999'
        assert 'generate' in dir(data_generator) and FLAGS.datasource == 'sinusoid'
        variances, errors, amps, phases = [], [], [], []
        data_generator.phase_range = [0, 2*np.pi]
        for _ in range(int(NUM_TEST_POINTS/2)):

            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            inputa = batch_x[:, :train_examples, :]
            inputb = batch_x[:,train_examples:, :]
            labela = batch_y[:, :train_examples, :]
            labelb = batch_y[:, train_examples:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

           # start with 5 grad steps only. Next try more than 5.
            saver.restore(sess, model_file1)
            result1 = sess.run([model.outputbs, model.total_losses2], feed_dict)
            saver.restore(sess, model_file2)
            result2 = sess.run([model.outputbs, model.total_losses2], feed_dict)
            saver.restore(sess, model_file3)
            result3 = sess.run([model.outputbs, model.total_losses2], feed_dict)
            saver.restore(sess, model_file4)
            result4 = sess.run([model.outputbs, model.total_losses2], feed_dict)
            saver.restore(sess, model_file5)
            result5 = sess.run([model.outputbs, model.total_losses2], feed_dict)
            results = [result1, result2, result3, result4, result5]

            # might want to measure median task variance instead of mean...
            variance = np.mean(np.var(np.squeeze(np.array([result[0][-1] for result in results])), axis = 0))
            error = result1[1][-1]
            variances.append(variance)
            errors.append(error)
            amps.append(amp[0])
            phases.append(phase[0])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        phases, variances, errors = np.asarray(phases), np.array(variances), np.array(errors)
        indistr_inds = np.nonzero(phases < np.pi)
        outdistr_inds = np.nonzero(phases >= np.pi)
        plt.scatter(variances[outdistr_inds], errors[outdistr_inds], c='r', marker = 'x')
        plt.scatter(variances[indistr_inds], errors[indistr_inds], c='b', marker ='+')

        plt.xlabel('mean task variance')
        plt.ylabel('20-shot mean task error')
        plt.title('Blue: in distribution, Red: out of distribution')
        import pdb; pdb.set_trace()
        plt.savefig('/home/cfinn/20gradstep_ensemble.png')

        return

    for test_point_i in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            if FLAGS.datasource == 'polynomial':
              batch_x, batch_y, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = data_generator.generate(train=False)
            else:
              batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'online' or FLAGS.baseline == 'incl_task':
                if FLAGS.datasource == 'sinusoid':
                    if FLAGS.amp_only:
                      batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 1])], 2)
                      batch_x[0, :, 1] = amp[0]
                    else:
                      batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                      batch_x[0, :, 1] = amp[0]
                      batch_x[0, :, 2] = phase[0]
                elif 'polynomial' in FLAGS.datasource: # siamese already has task id encoded in the input
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 10])], 2)
                    batch_x[0, :, 1] = d1[0]
                    batch_x[0, :, 2] = d2[0]
                    batch_x[0, :, 3] = d3[0]
                    batch_x[0, :, 4] = d4[0]
                    batch_x[0, :, 5] = d5[0]
                    batch_x[0, :, 6] = d6[0]
                    batch_x[0, :, 7] = d7[0]
                    batch_x[0, :, 8] = d8[0]
                    batch_x[0, :, 9] = d9[0]
                    batch_x[0, :, 10] = d10[0]

            inputa = batch_x[:, :train_examples, :]
            inputb = batch_x[:, train_examples:, :]
            labela = batch_y[:, :train_examples, :]
            labelb = batch_y[:, train_examples:, :]

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
                    #result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
                    result = sess.run(model.metaval_total_accuraciesa + model.metaval_total_accuracies2, feed_dict)
                else:
                    #result = sess.run([model.total_accuracy1] + model.total_accuracies2, feed_dict)
                    result = sess.run([model.total_accuracy1] + model.total_accuraciesa + model.total_accuracies2, feed_dict)
            else:  # this is for sinusoid
                #import pdb; pdb.set_trace()  # ask for model.outputas
                # saver.restore(sess, model_file)
                result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
            metaval_accuracies.append(result)
    #import pdb; pdb.set_trace()

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
            test_num_updates = 100  # normally 10
    else:
        assert not FLAGS.inner_sgd # not currently supported.
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            if FLAGS.train == True:
                test_num_updates = 10
            else:
                test_num_updates = 100 # 50
    if not FLAGS.train:
        test_num_updates = 20

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid' or FLAGS.datasource == 'polynomial' or 'omniglot' in FLAGS.datasource:
        if FLAGS.inner_sgd:
            num_updates = max(FLAGS.num_updates, test_num_updates)
            data_generator = DataGenerator(FLAGS.update_batch_size*(num_updates + 1), FLAGS.meta_batch_size)
        else:
            data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        assert FLAGS.datasource == 'miniimagenet'
        if FLAGS.metatrain_iterations == 0:
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.train:
                data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output
    #delete = data_generator.generate()
    if FLAGS.baseline == 'oracle' or FLAGS.baseline == 'online' or FLAGS.baseline == 'incl_task':
        assert FLAGS.datasource == 'sinusoid' or 'siamese' in FLAGS.datasource or 'mnist' in FLAGS.datasource or FLAGS.datasource == 'polynomial'
        if FLAGS.datasource == 'sinusoid':
            if FLAGS.amp_only:
                dim_input = 2
            else:
                dim_input = 3
        elif FLAGS.datasource == 'polynomial':
            dim_input = 11
        else:
            dim_input = data_generator.dim_input
        if FLAGS.baseline == 'oracle':
            FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
            FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    #if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot': # not including siamese omniglot
    if 'generate' not in dir(data_generator):
        assert not FLAGS.inner_sgd
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

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr) +'.metalr' + str(FLAGS.meta_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.fc_hidden != 40:
        exp_string += 'hidden' + str(FLAGS.fc_hidden)
    if FLAGS.num_fc != 2:
        exp_string += 'numfc' + str(FLAGS.num_fc)
    if FLAGS.fc_linear != 0:
        exp_string += 'fclin' + str(FLAGS.fc_linear)
    exp_string += ''


    if FLAGS.max_ent:
        exp_string += 'maxent'
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.l1_loss:
        exp_string += 'l1loss'
    if FLAGS.learned_loss:
        exp_string += 'learned_loss'
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

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        print(model_file)
        #model_file = model_file[:-3] + '000'
        #import pdb; pdb.set_trace()
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr, test_num_updates)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
