""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

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
        elif 'siamese' in FLAGS.datasource: # includes siamese_omniglot
            self.generate = self.generate_omniglot_batch
            self.num_classes = 1  # by default 1 (only relevant for classification problems)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)*2  # two images passed in.
            self.dim_output = 1
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_train = config.get('num_train', 1200)
            self.metatrain_character_folders = character_folders[:num_train]
            self.metaval_character_folders = character_folders[num_train:]
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
        elif 'rainbow_mnist' in FLAGS.datasource:
            # number of classes should be set to 1 for rainbow_mnist ( but dim output is 10 )
            self.num_classes = 1 
            assert FLAGS.num_classes == 1
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = 10 #self.num_classes
            self.generate = self.generate_rainbow_mnist_batch
            metatrain_folder = config.get('data_folder', '/home/cfinn/rainbow_mnist20b/train')
            metaval_folder = config.get('data_folder', '/home/cfinn/rainbow_mnist20b/val')

            self.metatrain_task_folders = [os.path.join(metatrain_folder, family) \
                for family in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, family)) ]
            random.seed(1)
            random.shuffle(self.metatrain_task_folders)
            self.metaval_task_folders = [os.path.join(metaval_folder, family) \
                for family in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, family)) ]
            self.rotations = config.get('rotations', [0])
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
            end_dir = tfolder[tfolder.rfind('/')+1:]
            first_imagefilepaths = None
            train_dirs = [end_dir in fold for fold in self.metatrain_task_folders]
            first_image_filepaths = None
            if np.any(train_dirs):
                first_train_dir_id = [i for i, x in enumerate(train_dirs) if x][0]
                first_train_dir = self.metatrain_task_folders[first_train_dir_id]
                first_image_filepaths = [os.path.join(first_train_dir, family, img_name) \
                    for family in os.listdir(first_train_dir) \
                    for img_name in os.listdir(os.path.join(first_train_dir, family))]

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
            scalar_labels = [int(sampled_filepaths[i][-12]) for i in range(self.num_samples_per_task)]
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


