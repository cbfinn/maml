""" Code for loading data. """
import imageio
import glob
import numpy as np
import os
import random
import pickle
import tensorflow as tf
from collections import defaultdict

from skimage import transform

from tensorflow.python.platform import flags
from tensorflow.examples.tutorials.mnist import input_data
from utils import get_images, load_transform, load_transform_color

FLAGS = flags.FLAGS


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_task, batch_size, config={}):
        """
        Args:
            num_samples_per_task: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_task = num_samples_per_task
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.dim_state_input = 0 #None

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            if FLAGS.train:
                self.amp_range = config.get('amp_range', [0.1, 5.0])
            else:
                self.amp_range = config.get('amp_range', [5.0, 10.0])
                #self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            #self.amp_range = config.get('amp_range', [0.0, 20.0])
            #self.phase_range = config.get('phase_range', [0, 2*np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif 'cifar' in FLAGS.datasource: # includes siamese_cifar
            #self.generate = self.generate_omniglot_batch
            assert 'cont' in FLAGS.datasource
            self.generate = self.generate_cont_cifar_batch
            self.num_classes = 1  # by default 1 (only relevant for classification problems)
            self.img_size = config.get('img_size', (32, 32))
            self.dim_input = np.prod(self.img_size)*2*3  # two color images passed in.
            self.dim_output = 1
            # data that is pre-resized using PIL with lanczos filter
            self.data_folder = config.get('data_folder', './data/cifar-100-python')
            self.task_folders = list(range(100))
            random.seed(1)
            random.shuffle(self.task_folders) 
            # dataset has 24 batches per task, each batch has one image
            self.cur_task = 49 # current task, indexes into self.task_folders
            self.cur_task_batch_id = 10  # number of batches for the current task
            self.num_tasks = 50   # total number of tasks with data so far
            self.task_data = defaultdict(list)
            for i in range(self.num_tasks):
                if i < self.cur_task:
                    self.task_data[i] = list(range(100)) #24*20))
                else:
                    self.task_data[i] = list(range(24*(self.cur_task_batch_id+1)))
 
            #self.task_data[self.cur_task].extend(list(range(self.cur_task_batch_id*12, (self.cur_task_batch_id+1)*12)))
            #self.cur_task_batch_id += 1
            #character_folders = [os.path.join(data_folder, family, character) \
            #    for family in os.listdir(data_folder) \
            #    if os.path.isdir(os.path.join(data_folder, family)) \
            #    for character in os.listdir(os.path.join(data_folder, family))]
            #random.seed(1)
            #random.shuffle(character_folders)
            #num_train = config.get('num_train', 1200)
            #self.metatrain_character_folders = character_folders[:num_train]
            #self.metaval_character_folders = character_folders[num_train:]
            self.load_cifar()
        elif 'mnist' in FLAGS.datasource and 'rainbow' not in FLAGS.datasource:
            self.generate = self.generate_mnist_batch
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = 10
            self.T = np.array([[1, 0,-14],[0, 1,-14],[0, 0, 1]])
            self.invT = np.linalg.inv(self.T)
            self.mnist = input_data.read_data_sets('/home/cfinn/mnist_data', one_hot=True)
            #self.num_classes = 10
            #self.mnist = input_data.read_data_sets('/home/cfinn/mnist_data', one_hot=False)

        elif 'omniglot' in FLAGS.datasource:
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if FLAGS.test_set:
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            else:
                self.metaval_character_folders = character_folders[num_train+num_val:]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif 'push' in FLAGS.datasource:   
            #assert FLAGS.update_batch_size == 20 # TODO - update batch size of 20?
            self.img_size = config.get('img_size', 125) 
            self.dim_input = self.img_size*self.img_size* 3
            self.dim_state_input = 20
            self.dim_output = 7 #self.num_classes
            self.generate = self.generate_cont_push_batch 
            self.data_folder = '/media/4tb/cfinn/paired_consistent_push_demos/'
            self.tasks = [int(filename[filename.rfind('/')+1:-4]) for filename in glob.glob(self.data_folder + '*.pkl')]
            # dataset has 12 batches per task, each batch has 1 demonstration
            self.cur_task = 99 #49 # current task, indexes into self.task_folders
            self.cur_task_batch_id = 10  # number of batches for the current task
            self.num_tasks = 100 #50   # total number of tasks with data so far
            self.task_data = {}
            for i in range(self.num_tasks):
                self.task_data[i] = list(range(self.cur_task_batch_id))
            self.load_pushing()
        elif 'pascal' in FLAGS.datasource:
            self.num_classes = 1 
            self.img_size = config.get('img_size', (125, 125))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = 4 #self.num_classes
            self.rotations = config.get('rotations', [0])
            self.generate = self.generate_cont_pascal_batch
            data_folder = config.get('data_folder', '/home/cfinn/fixedtexture_norb_pascal1000')
            self.task_folders = [os.path.join(data_folder, task) for task in os.listdir(data_folder)]
            random.seed(1)
            random.shuffle(self.task_folders)
            self.cur_task = 19 # current task, indexes into self.task_folders
            self.cur_task_batch_id = 0  # number of batches for the current task
            self.num_tasks = 20   # total number of tasks with data so far
            self.task_data = defaultdict(list)
            for i in range(self.num_tasks):
                if i < self.cur_task:
                    self.task_data[i] = list(range(1000)) #24*20))
                else:
                    self.task_data[i] = list(range(20))
                    #self.task_data[i] = list(range(500))
            self.load_pascal()

        elif 'rainbow_mnist' in FLAGS.datasource:
            # number of classes should be set to 1 for rainbow_mnist ( but dim output is 10 )
            self.num_classes = 1 
            assert FLAGS.num_classes == 1
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = 10 #self.num_classes
            self.rotations = config.get('rotations', [0])
            if 'cont' in FLAGS.datasource:
                self.generate = self.generate_cont_rainbow_mnist_batch
                data_folder = config.get('data_folder', '/home/cfinn/' + FLAGS.datasource +'/')
                # this will be continually added to, but it will start with 8 folders and 10 batches per folder
                self.task_folders = [os.path.join(data_folder, task) for task in os.listdir(data_folder)]
                random.seed(1)
                random.shuffle(self.task_folders)

                # cur task = 0, num_tasks = 1 for 0
                # cur task = 19, num_tasks = 20 for 0
                self.cur_task = 19 # current task, indexes into self.task_folders
                self.cur_task_batch_id = 10  # number of batches for the current task
                self.num_tasks = 20   # total number of tasks with data so far
                self.task_data = {}
                for i in range(self.num_tasks):
                    self.task_data[i] = list(range(self.cur_task_batch_id))
                self.load_rainbow_mnist()
            else:
                self.generate = self.generate_rainbow_mnist_batch
                metatrain_folder = config.get('data_folder', '/home/cfinn/' + FLAGS.datasource +'/train')
                metaval_folder = config.get('data_folder', '/home/cfinn/' + FLAGS.datasource + '/val')

                self.metatrain_task_folders = [os.path.join(metatrain_folder, family) \
                    for family in os.listdir(metatrain_folder) \
                    if os.path.isdir(os.path.join(metatrain_folder, family)) ]
                random.seed(1)
                random.shuffle(self.metatrain_task_folders)
                self.metaval_task_folders = [os.path.join(metaval_folder, family) \
                    for family in os.listdir(metaval_folder) \
                    if os.path.isdir(os.path.join(metaval_folder, family)) ]
        elif FLAGS.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/train')
            if FLAGS.test_set:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')

    def load_pushing(self):
        print('Loading data into RAM')
        self.images = {}
        self.pkls = {}
        # load 100 tasks into memory
        tasks = list(enumerate(self.tasks))[:150]
        for task_index, task in tasks:
            with open(self.data_folder + str(task) + '.pkl', 'rb') as pkl_file:
                self.pkls[self.data_folder+str(task) + '.pkl'] = pickle.load(pkl_file)
            for demo in range(12, 24):
                filename = self.data_folder + 'object_' + str(task) + '/cond' + str(demo) + '.samp0.gif'
                self.images[filename] = np.array(imageio.mimread(filename))[:, :, :, :3]
        print('Done loading images')
    
    def load_cifar(self):
        with open(self.data_folder + '/train', 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        images = data_dict[b'data']
        images = np.reshape(images, [-1, 3, 32, 32])
        images = np.transpose(images, [0, 2, 3, 1])
        labels = data_dict[b'fine_labels']
        self.image_dict = defaultdict(list)
        for i, label in enumerate(labels):
            self.image_dict[label].append(images[i].astype('float32')/255.)
        for key in self.image_dict.keys():
            self.image_dict[key] = np.array(self.image_dict[key])
        del images
 
    def load_pascal(self):
        print('Loading images into RAM')
        self.images = {}
        self.task_images = {}
        for tfolder in self.task_folders:
            filepaths = glob.glob(tfolder+'/*.png')
            self.task_images[tfolder] = filepaths
            for filepath in filepaths:
               self.images[filepath] = load_transform_color(filepath, size=self.img_size)
        print('Done loading images')

    def load_rainbow_mnist(self):
        print('Loading images into RAM')
        self.images = {}
        for tfolder in self.task_folders:
           image_filepaths = [os.path.join(tfolder, batch, family, img_name) \
                for batch in os.listdir(tfolder) \
                for family in os.listdir(os.path.join(tfolder, str(batch))) \
                for img_name in os.listdir(os.path.join(tfolder, str(batch), family))]
           for filepath in image_filepaths:
               self.images[filepath] = load_transform_color(filepath, size=self.img_size)
        print('Done loading images')
 
    # CIFAR - 500 images per task, 40 batches per task
    def add_task(self):
        # Performance on current task is satisfactory. Move on to next task.
        assert 'cont' in FLAGS.datasource
        self.cur_task += 1
        if self.cur_task >= self.num_tasks:
            self.num_tasks += 1
            if 'cifar' in FLAGS.datasource or 'pascal' in FLAGS.datasource:
                self.cur_task_batch_id = 0 
                self.add_batch()
            else:
                self.cur_task_batch_id = 1
                self.task_data[self.cur_task] = [0]
        else:
            self.cur_task_batch_id = 10 

    # for cifar, add 12 images (batches don't exist)
    def add_batch(self):
        # Need to add more data for the current task
        assert 'cont' in FLAGS.datasource
        if 'cifar' in FLAGS.datasource:
            self.task_data[self.cur_task].extend(list(range(self.cur_task_batch_id*24, (self.cur_task_batch_id+1)*24)))
            self.cur_task_batch_id += 1
        elif 'pascal' in FLAGS.datasource:
            self.task_data[self.cur_task].extend(list(range(self.cur_task_batch_id*20, (self.cur_task_batch_id+1)*20)))
            self.cur_task_batch_id += 1
        else:
            self.task_data[self.cur_task].append(self.cur_task_batch_id)
            self.cur_task_batch_id += 1

    def generate_cont_cifar_batch(self, train=True, itr=None):  # RGB images
        if train:
            # Use all tasks so far
            if FLAGS.train_only_on_cur:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task]]
            else:
                task_folders = list(enumerate(self.task_folders))[:self.num_tasks]
        else:
            # Only use the next task # current task
            # cont_incl_cur: whether or not to meta-train on the current task
            if FLAGS.cont_incl_cur:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task]]
            else:
                # TODO - check this
                task_folders = [list(enumerate(self.task_folders))[self.cur_task+1]]
                task_folders[0] = (task_folders[0][0] - 1, task_folders[0][1])

        # if not train and not inner SGD, use more samples per task
        if not train and not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
            num_val = int((self.num_samples_per_task / 2.) * 3.0/2.0)
            task_index, _ = task_folders[0]
            num_left = len(self.task_data[task_index]) - num_val
            num_train = int(num_left * 2. / 3.)
            num_samples_per_task = num_val + num_train 
        else:
            num_samples_per_task = self.num_samples_per_task
            num_train = num_val = int(num_samples_per_task / 2)
        inputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_input], dtype=np.float32)
        outputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_output], dtype=np.int32)

        # sample tasks
        task_folders = [random.choice(task_folders) for _ in range(self.batch_size)]

        for batch_i in range(self.batch_size):
            task_index, tfolder = task_folders[batch_i]

            available_batches = self.task_data[task_index]
            if not train:
                last_batch = max(available_batches)
                available_batches = list(range(last_batch, min(last_batch+24,500)))

            if len(available_batches) < num_samples_per_task*3./2.:
                same_idx = np.random.choice(available_batches, size=int(num_samples_per_task*3./2.), replace=True)
            else:
                same_idx = np.random.choice(available_batches, size=int(num_samples_per_task*3./2.), replace=False)

            ref_images = self.image_dict[tfolder][same_idx[:num_samples_per_task]]  # all reference images
            comp_images1 = self.image_dict[tfolder][same_idx[num_samples_per_task:]]   # all same comparison images
            # sample images that are from different classes
            choices = list(range(self.cur_task+1))
            choices.pop(task_index)
            other_classes_idx = np.random.choice(choices, size=int(num_samples_per_task/2), replace=True)
            other_classes = [self.task_folders[o_class] for o_class in other_classes_idx]
            comp_images_idx = [np.random.choice(self.task_data[index]) for index in other_classes_idx]
            # all different comparison images
            comp_images2 = np.array([self.image_dict[img_class][index] for img_class, index in zip(other_classes, comp_images_idx)])

            image_pairs_same = np.concatenate([ref_images[:int(num_samples_per_task/2)], comp_images1], 3)           
            image_pairs_diff = np.concatenate([ref_images[int(num_samples_per_task/2):], comp_images2], 3)           

            # figure out num_train, num_val
            if not train and not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all: 
                nthalf = int(num_train/2)
                same1 = image_pairs_same[:nthalf]
                diff1 = image_pairs_diff[:nthalf]
                same2 = image_pairs_same[nthalf:]
                diff2 = image_pairs_diff[nthalf:]
                labels_same1 = np.ones([nthalf, 1])
                labels_diff1 = np.zeros([nthalf, 1])
                labels_same2 = np.ones([int(num_val/2), 1])
                labels_diff2 = np.zeros([int(num_val/2), 1])
                label_batch = np.concatenate([labels_same1, labels_diff1, labels_same2, labels_diff2], 0)
            else:
                same1, same2 = np.split(image_pairs_same, 2, 0)
                labels_same = np.ones([int(num_samples_per_task/4),1])
                diff1, diff2 = np.split(image_pairs_diff, 2, 0)
                labels_diff = np.zeros([int(num_samples_per_task/4),1])
                label_batch = np.concatenate([labels_same, labels_diff, labels_same, labels_diff], 0)

            # make pre update and post update consistent
            img_batch = np.concatenate([same1, diff1, same2, diff2], 0)

            assert not FLAGS.inner_sgd # not currently supported

            inputs[batch_i] = img_batch.reshape([num_samples_per_task, -1])
            outputs[batch_i] = label_batch
        return inputs, outputs, None, None


    def generate_cont_push_batch(self, train=True, itr=None):
        if train:
            if FLAGS.train_only_on_cur:
                tasks = [list(enumerate(self.tasks))[self.cur_task]]
            else:
                tasks = list(enumerate(self.tasks))[:self.num_tasks]
        else:
            if FLAGS.cont_incl_cur:
                tasks = [list(enumerate(self.tasks))[self.cur_task]]
            else:
                tasks = [list(enumerate(self.tasks))[self.cur_task+1]]

        # if not train and not inner SGD, use more samples per task
        if not train and not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
            eval_samples_per_task = int(self.num_samples_per_task/2)
            # update batch size * the number of available batches
            num_train = int(self.num_samples_per_task / 2 * len(self.task_data[self.cur_task]))
            num_samples_per_task = num_train + eval_samples_per_task
        else:
            num_samples_per_task = self.num_samples_per_task

        inputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_input], dtype=np.float32)
        state_inputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_state_input], dtype=np.float32)
        outputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_output], dtype=np.float32)

        tasks = [random.choice(tasks) for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            task_index, task = tasks[i]
            available_batches = np.array(self.task_data[task_index]) + 12
            if not FLAGS.cont_incl_cur:
                # not fully supported, need to update available_batches
                import pdb; pdb.set_trace()
            val_batches = None
            if not train and FLAGS.baseline != 'oracle':
                val_batches = [available_batches[-1] + 1]
            elif not train and FLAGS.baseline == 'oracle':
                available_batches = [available_batches[-1] + 1]
            assert not FLAGS.inner_sgd
            # first sample one demo 
            if val_batches is None:
                demo_inds = np.random.choice(available_batches, size=2, replace=True)
            elif val_batches is not None and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
                val_ind = np.random.choice(val_batches)
            else:
                demo_inds = [np.random.choice(available_batches), np.random.choice(val_batches)]
            # load pickle file
            pkl = self.pkls[self.data_folder + str(task) + '.pkl']
            states = pkl['demoX']
            vel = False
            if vel:     
                diff = states[:,1:,:7] - states[:,:-1,:7]
                actions = np.concatenate([diff, np.expand_dims(states[:,-1,7:14], 1)], 1)
            else:
                actions = pkl['demoU']  
            if not train and not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
                all_actions = []
                all_states = []
                all_images = []
                for demo_batch in available_batches:
                    timesteps = np.random.choice(np.arange(100), size=int(self.num_samples_per_task/2), replace=False)
                    all_actions.append(actions[demo_batch, timesteps, :])
                    all_states.append(states[demo_batch, timesteps, :])
                    all_images.append(self.images[self.data_folder + 'object_' + str(task) + '/cond' + str(demo_batch) + '.samp0.gif'][timesteps, :, :, :])
                timesteps = np.random.choice(np.arange(100), size=int(self.num_samples_per_task/2), replace=False)
                all_actions.append(actions[val_ind, timesteps, :])
                all_states.append(states[val_ind, timesteps, :])
                all_images.append(self.images[self.data_folder + 'object_' + str(task) + '/cond' + str(val_ind) + '.samp0.gif'][timesteps, :, :, :])
                actions = np.concatenate(all_actions, 0)
                states = np.concatenate(all_states, 0)
                images = np.concatenate(all_images, 0)
            else:
                timesteps0 = np.random.choice(np.arange(100), size=int(num_samples_per_task/2), replace=False)
                timesteps1 = np.random.choice(np.arange(100), size=int(num_samples_per_task/2), replace=False)
                # select actions from pickle file
                actions = np.concatenate([actions[demo_inds[0], timesteps0, :], actions[demo_inds[1], timesteps1, :]], 0)
                states = np.concatenate([states[demo_inds[0], timesteps0, :], states[demo_inds[1], timesteps1, :]], 0)
                images0 = self.images[self.data_folder + 'object_' + str(task) + '/cond' + str(demo_inds[0]) + '.samp0.gif']
                images1 = self.images[self.data_folder + 'object_' + str(task) + '/cond' + str(demo_inds[1]) + '.samp0.gif']
                images = np.concatenate([images0[timesteps0, :, :, :],  images1[timesteps1, :, :, :]], 0)
            outputs[i] = actions
            state_inputs[i] = states
            inputs[i] = images.reshape([num_samples_per_task, -1])

        return inputs, outputs, state_inputs, None

    def generate_cont_pascal_batch(self, train=True, itr=None):  # RGB images
        if train:
            # Use all tasks so far
            if FLAGS.train_only_on_cur:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task]]
            else:
                task_folders = list(enumerate(self.task_folders))[:self.num_tasks]
        else:
            # Only use the next task # current task
            # cont_incl_cur: whether or not to meta-train on the current task
            if FLAGS.cont_incl_cur:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task]]
            else:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task+1]]
                task_folders[0] = (task_folders[0][0] - 1, task_folders[0][1])

        # if not train and not inner SGD, use more samples per task
        if not train and not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
            eval_samples_per_task = int(self.num_samples_per_task/2)
            task_index, tfolder = task_folders[0]
            available_batches = self.task_data[task_index]
            image_filepaths = np.array(self.task_images[tfolder])[np.array(available_batches)] 
            num_train = min(len(image_filepaths), 300)  # prevent OOM errors
            num_samples_per_task = num_train + eval_samples_per_task
        else:
            num_samples_per_task = self.num_samples_per_task
        inputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_input], dtype=np.float32)
        outputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_output], dtype=np.float32)

        # sample tasks
        task_folders = [random.choice(task_folders) for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            task_index, tfolder = task_folders[i]
            available_batches = self.task_data[task_index]
            # use self.task_data[task_index]  to figure out which folders can be used
            image_filepaths = np.array(self.task_images[tfolder])[np.array(available_batches)] 
            val_image_filepaths = None
            if not train and FLAGS.baseline != 'oracle':
                val_image_filepaths = self.task_images[tfolder][available_batches[-1]+1:] 
            elif not train and FLAGS.baseline == 'oracle':
                image_filepaths = self.task_images[tfolder][available_batches[-1]+1:] 
            assert not FLAGS.shuffle_tasks 
            assert not FLAGS.inner_sgd # not currently supported

            # sample num_samples_per_task images, num_samples_per_task should be <= 5
            if val_image_filepaths is not None:
                assert not FLAGS.inner_sgd # not supported
                sampled_filepaths = np.random.choice(image_filepaths, size=num_samples_per_task-int(self.num_samples_per_task/2), replace=False)
                second_filepaths = np.random.choice(val_image_filepaths, size=int(self.num_samples_per_task/2), replace=False)
                sampled_filepaths = np.concatenate([sampled_filepaths, second_filepaths])
            else:
                sampled_filepaths = np.random.choice(image_filepaths, size=num_samples_per_task, replace=False)
            #images = [np.reshape(np.array(load_transform_color(filename, size=self.img_size)), (-1)) for filename in sampled_filepaths]
            images = [np.reshape(np.array(self.images[filename]), (-1)) for filename in sampled_filepaths]
            inputs[i] = np.array(images)

            # extract label, convert to one-hot
            img_names = [sampled_filepaths[i][sampled_filepaths[i].rfind('/')+1:-4] for i in range(num_samples_per_task)]
            labels = [np.array(img_name.split('_')) for img_name in img_names]
            labels = np.float32(np.array(labels))
            sin_cos = np.array([[np.sin(angle), np.cos(angle)] for angle in labels[:,-1]])
            outputs[i, :, :2] = labels[:,:2]
            outputs[i, :, 2:] = sin_cos
            
        return inputs, outputs, None, None




    def generate_cont_rainbow_mnist_batch(self, train=True, itr=None):  # RGB images

        if train:
            # Use all tasks so far
            if FLAGS.train_only_on_cur:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task]]
            else:
                task_folders = list(enumerate(self.task_folders))[:self.num_tasks]
        else:
            # Only use the next task # current task
            # cont_incl_cur: whether or not to meta-train on the current task
            if FLAGS.cont_incl_cur:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task]]
            else:
                task_folders = [list(enumerate(self.task_folders))[self.cur_task+1]]
                task_folders[0] = (task_folders[0][0] - 1, task_folders[0][1])

        # if not train and not inner SGD, use more samples per task
        if not train and not FLAGS.inner_sgd and FLAGS.baseline != 'oracle' and FLAGS.cont_finetune_on_all:
            eval_samples_per_task = int(self.num_samples_per_task/2)
            task_index, tfolder = task_folders[0]
            available_batches = self.task_data[task_index]
            image_filepaths = [os.path.join(tfolder, str(batch), family, img_name) \
                for batch in available_batches \
                for family in os.listdir(os.path.join(tfolder, str(batch))) \
                for img_name in os.listdir(os.path.join(tfolder, str(batch), family))]
            num_train = len(image_filepaths)
            num_samples_per_task = num_train + eval_samples_per_task
        else:
            num_samples_per_task = self.num_samples_per_task
        inputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_input], dtype=np.float32)
        outputs = np.zeros([self.batch_size, num_samples_per_task, self.dim_output], dtype=np.int32)

        # sample tasks
        task_folders = [random.choice(task_folders) for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            task_index, tfolder = task_folders[i]
            available_batches = self.task_data[task_index]
            # use self.task_data[task_index]  to figure out which folders can be used
            image_filepaths = [os.path.join(tfolder, str(batch), family, img_name) \
                for batch in available_batches \
                for family in os.listdir(os.path.join(tfolder, str(batch))) \
                for img_name in os.listdir(os.path.join(tfolder, str(batch), family))]
            val_image_filepaths = None
            if not train and FLAGS.baseline != 'oracle':
                batch = available_batches[-1] + 1
                val_image_filepaths = [os.path.join(tfolder, str(batch), family, img_name)
                    for family in os.listdir(os.path.join(tfolder, str(batch))) \
                    for img_name in os.listdir(os.path.join(tfolder, str(batch), family))]
            elif not train and FLAGS.baseline == 'oracle':
                batch = available_batches[-1] + 1
                image_filepaths = [os.path.join(tfolder, str(batch), family, img_name)
                    for family in os.listdir(os.path.join(tfolder, str(batch))) \
                    for img_name in os.listdir(os.path.join(tfolder, str(batch), family))]
            assert not FLAGS.shuffle_tasks 
            assert not FLAGS.inner_sgd # not currently supported

            # sample num_samples_per_task images, num_samples_per_task should be <= 5
            if val_image_filepaths is not None:
                assert not FLAGS.inner_sgd # not supported
                sampled_filepaths = np.random.choice(image_filepaths, size=num_samples_per_task-int(self.num_samples_per_task/2), replace=False)
                second_filepaths = np.random.choice(val_image_filepaths, size=int(self.num_samples_per_task/2), replace=False)
                sampled_filepaths = np.concatenate([sampled_filepaths, second_filepaths])
            else:
                sampled_filepaths = np.random.choice(image_filepaths, size=num_samples_per_task, replace=False)
            #images = [np.reshape(np.array(load_transform_color(filename, size=self.img_size)), (-1)) for filename in sampled_filepaths]
            images = [np.reshape(np.array(self.images[filename]), (-1)) for filename in sampled_filepaths]
            inputs[i] = np.array(images)

            # extract label, convert to one-hot
            scalar_labels = [int(sampled_filepaths[i][sampled_filepaths[i].rfind('/')-1]) for i in range(num_samples_per_task)]
            outputs[i, np.arange(num_samples_per_task), scalar_labels] = 1.0
        return inputs, outputs, None, None



    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_task, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource == 'miniimagenet':
            image = tf.image.decode_jpeg(image_file)
            image.set_shape((self.img_size[0],self.img_size[1],3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0],self.img_size[1],1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_task
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_task):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_task + k
                new_list.append(tf.gather(image_batch,true_idxs))
                if FLAGS.datasource == 'omniglot': # and FLAGS.train:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
                        k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_task, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def get_sinusoid_amp_range(self, itr, train=True):
        if FLAGS.datadistr == 'stationary':
            if train:
                return self.amp_range 
            else:
                return [5.0,10.0]
        elif FLAGS.datadistr == 'continual5':
            if train == True:
              if itr < 10000: # 8k
                return [0.0, 5.0]
              elif itr < 20000:
                return [0.0, 7.0]
              elif itr < 30000:
                return [0.0, 9.0]
              elif itr < 40000:
                return [0.0, 11.0]
              elif itr < 50000:
                return [0.0, 13.0]
              #elif itr < 20000:
                #return [0.0, 15.0]
              #elif itr < 22000:
                #return [0.0, 17.0]
              else:
                raise NotImplementedError('done training')
            else:
              if itr < 10000:
                return [0.0, 5.0]
              elif itr < 20000:
                return [5.0, 7.0]
              elif itr < 30000:
                return [7.0, 9.0]
              elif itr < 40000:
                return [9.0, 11.0]
              elif itr < 50000:
                return [11.0, 13.0]
              #elif itr < 20000:
                #return [13.0, 15.0]
              #elif itr < 22000:
                #return [15.0, 17.0]
              else:
                raise NotImplementedError('done training')

    def generate_sinusoid_batch(self, itr=-1, train=True, input_idx=None):
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp_range = self.get_sinusoid_amp_range(itr=itr, train=train)
        amp = np.random.uniform(amp_range[0], amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_task, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_task, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_task, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_task-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase

    def sample_img_transform(self):
        scale_range = [1,3] #2]
        rotation_range = [-np.pi, np.pi] #/2, np.pi/2]
        shear_range = [-np.pi/4.0, np.pi/4.0]
        sc = np.random.uniform(scale_range[0], scale_range[1], (2))
        rot = np.random.uniform(rotation_range[0], rotation_range[1])
        sh = np.random.uniform(shear_range[0], shear_range[1])
        tform = transform.AffineTransform(scale=sc, rotation=rot, shear=sh)
        new_params = self.invT.dot(tform.params).dot(self.T)
        tform = transform.AffineTransform(new_params)
        return tform

    """
    def generate_mnist_batch(self, train=True):
        #from tensorflow.examples.tutorials.mnist import input_data
        #mnist = input_data.read_data_sets(self.data_folder, one_hot=False)
        mnist = self.mnist

        inputs = np.zeros([self.batch_size, self.num_classes * self.num_samples_per_task, self.dim_input], dtype=np.float32)
        outputs = np.zeros([self.batch_size, self.num_classes * self.num_samples_per_task, self.dim_output], dtype=np.int32)

        assert self.num_classes == 10

        for i in range(self.batch_size):
            first_set, others = [], []
            for j in range(self.num_classes):
                #examples =  np.nonzero(np.all(mnist.train.labels == np.array([int(ind == j) for ind in range(10)]), axis=1))
                examples =  np.nonzero(mnist.train.labels == j)
                inds = np.random.choice(examples[0], size=self.num_samples_per_task, replace=False)
                first_set.append(inds[0])
                others.extend(inds[1:])
            random.shuffle(first_set)
            random.shuffle(others)
            indices = first_set + others
            inputs[i] = mnist.train.images[indices,:] # maybe invert
            outputs[i, np.arange(self.num_classes*self.num_samples_per_task), mnist.train.labels[indices]] = 1.0
        return inputs, outputs, None, None

    """
    def generate_mnist_batch(self, train=True, itr=None):

        batch_size = self.batch_size
        inputs = np.zeros([batch_size, self.num_samples_per_task, self.dim_input], dtype=np.float32)
        outputs = np.zeros([batch_size, self.num_samples_per_task, self.dim_output], dtype=np.int32)

        if train:
            imgs, labels = self.mnist.train.next_batch(self.num_samples_per_task*batch_size) 
        else:
            imgs, labels = self.mnist.validation.next_batch(self.num_samples_per_task*batch_size) 
        for i in range(batch_size):
            begin = i*self.num_samples_per_task
            end = (i+1)*self.num_samples_per_task
            task_imgs = imgs[begin:end]
            if FLAGS.incl_switch and random.uniform(0,1) > 0.5:
                task_imgs_reshaped = np.reshape(task_imgs, [self.num_samples_per_task,28,28])
                # switch top and bottom of image (or left and right?)
                task_imgs = np.reshape(np.concatenate([task_imgs_reshaped[:,14:,:], task_imgs_reshaped[:,:14,:]], 1) , [self.num_samples_per_task, -1]) 
            task_labels = labels[begin:end]
            task_tform = self.sample_img_transform()
            tformed = np.array([np.reshape(transform.warp(np.reshape(img, (28,28,1)),task_tform), (-1)) for img in task_imgs])
            inputs[i] = tformed
            outputs[i] = task_labels
        return inputs, outputs, None, None


    def generate_rainbow_mnist_batch(self, train=True, itr=None):  # RGB images

        inputs = np.zeros([self.batch_size, self.num_samples_per_task, self.dim_input], dtype=np.float32)
        outputs = np.zeros([self.batch_size, self.num_samples_per_task, self.dim_output], dtype=np.int32)

        #assert self.num_samples_per_task <= 5
        #assert self.num_classes == 10

        if train:
            folders = self.metatrain_task_folders
        else:
            folders = self.metaval_task_folders
        # sample tasks
        task_folders = [random.choice(folders) for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            tfolder = task_folders[i]

            first_image_filepaths = None
            end_dir = tfolder[tfolder.rfind('/')+1:]
            train_dirs = [end_dir in fold for fold in self.metatrain_task_folders]
            if np.any(train_dirs) and train == False:
                first_train_dir_id = [i for i, x in enumerate(train_dirs) if x][0]
                first_train_dir = self.metatrain_task_folders[first_train_dir_id]
                first_image_filepaths = [os.path.join(first_train_dir, family, img_name) \
                    for family in os.listdir(first_train_dir) \
                    for img_name in os.listdir(os.path.join(first_train_dir, family))]

            if FLAGS.shuffle_tasks:
                image_filepaths = [os.path.join(folder, family, img_name) \
                    for folder  in folders \
                    for family in os.listdir(folder) \
                    for img_name in os.listdir(os.path.join(folder, family))]
            else:
                image_filepaths = [os.path.join(task_folders[i], family, img_name) \
                    for family in os.listdir(task_folders[i]) \
                    for img_name in os.listdir(os.path.join(task_folders[i], family))]
            # sample num_samples_per_task images, num_samples_per_task should be <= 5
            if first_image_filepaths is not None:
                assert not FLAGS.inner_sgd # not supported
                sampled_filepaths = np.random.choice(first_image_filepaths, size=int(self.num_samples_per_task/2), replace=False)
                second_filepaths = np.random.choice(image_filepaths, size=int(self.num_samples_per_task/2), replace=False)
                sampled_filepaths = np.concatenate([sampled_filepaths, second_filepaths])
            else:
                sampled_filepaths = np.random.choice(image_filepaths, size=self.num_samples_per_task, replace=False)
            images = [np.reshape(np.array(load_transform_color(filename, size=self.img_size)), (-1)) for filename in sampled_filepaths]
            inputs[i] = np.array(images)

            # extract label, convert to one-hot
            scalar_labels = [int(sampled_filepaths[i][sampled_filepaths[i].rfind('/')-1]) for i in range(self.num_samples_per_task)]
            outputs[i, np.arange(self.num_samples_per_task), scalar_labels] = 1.0
        return inputs, outputs, None, None



    def generate_omniglot_batch(self, train=True):
        ### Siamese omniglot comparison task.
        batch_size = self.batch_size
        inputs = np.zeros([batch_size, self.num_samples_per_task, self.dim_input], dtype=np.float32)
        all_input_ids = []
        outputs = np.zeros([batch_size, self.num_samples_per_task, self.dim_output], dtype=np.int32)

        if train:
            folders = self.metatrain_character_folders
        else:
            folders = self.metaval_character_folders
        #families = list(folders.keys())
        #sampled_families = [random.choice(families) for _ in range(batch_size)]  # chosen alphabets
        sampled_characters = [random.choice(folders) for _ in range(batch_size)]  # chosen alphabets

        prob_same = 0.3
        #num_same = int(prob_same*batch_size)

        #for fam_i, fam in enumerate(sampled_families):  # meta batch
        for char_i, char in enumerate(sampled_characters):  # meta batch
            for samp_i in range(self.num_samples_per_task):
              if random.uniform(0,1) < prob_same:
                #char_folders = random.sample(folders[fam], 1)[0]
                images = random.sample(os.listdir(char), 2)
                images = [char + '/' + image for image in images]
                label = 1
              else:
                #char_folders = random.sample(folders[fam], 2)
                #image_0 = random.sample(os.listdir(char_folders[0]), 1)
                #image_1 = random.sample(os.listdir(char_folders[1]), 1)
                image_0 = random.sample(os.listdir(char), 1)
                random_char = random.choice([f for f in folders if f != char])
                image_1 = random.sample(os.listdir(random_char), 1)
                images = [char + '/' + image_0[0], random_char + '/' + image_1[0]]
                label = 0
              images = [np.array(load_transform(filename, size=self.img_size)) for filename in images]
              images = np.array(np.stack(images, 2), dtype=np.float32).flatten()  # concat along channel dim
              inputs[char_i, samp_i] = images
              outputs[char_i, samp_i] = label

            # one-hot representation
            #outputs[i,np.arange(self.num_classes*self.num_samples_per_task),labels] = 1.0

        return inputs, outputs, None, None


