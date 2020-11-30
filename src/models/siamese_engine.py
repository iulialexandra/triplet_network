import os
import logging
import numpy.random as rng
import numpy as np
from tqdm import tqdm
import time
from signal import SIGINT, SIGTERM
import tensorflow as tf
import tools.utils as util
import tools.visualization as vis
import data_processing.dataset_utils as dat
from sklearn.manifold import TSNE
import keras.backend as K
from keras.utils import to_categorical
from collections import namedtuple
from collections import defaultdict
from contextlib import redirect_stdout
from keras.optimizers import Adam
from tools.modified_sgd import Modified_SGD
from networks.horizontal_nets import *
from networks.original_nets import *
from networks.resnets import *
from networks.triplet_nets import *
from models.triplet_loss import *

logger = logging.getLogger("siam_logger")


class SiameseEngine():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.evaluate_every = args.evaluate_every
        self.results_path = args.results_path
        self.model_name = args.model
        self.num_val_ways = args.num_val_ways
        self.num_shots = args.num_shots
        self.num_val_trials = args.num_val_trials
        self.image_dims = args.image_dims
        self.results_path = args.results_path
        self.plot_confusion = args.plot_confusion
        self.plot_training_images = args.plot_training_images
        self.plot_wrong_preds = args.plot_wrong_preds
        self.plot_val_images = args.plot_val_images
        self.plot_test_images = args.plot_test_images
        self.learning_rate = args.learning_rate
        self.lr_annealing = args.lr_annealing
        self.momentum_annealing = args.momentum_annealing
        self.momentum_slope = args.momentum_slope
        self.final_momentum = args.final_momentum
        self.optimizer = args.optimizer
        self.triplet_strategy = args.triplet_strategy
        self.save_weights = args.save_weights
        self.checkpoint = args.checkpoint
        self.left_classif_factor = args.left_classif_factor
        self.right_classif_factor = args.right_classif_factor
        self.siamese_factor = args.siamese_factor
        self.write_to_tensorboard = args.write_to_tensorboard
        self.summary_writer = tf.summary.FileWriter(self.results_path)

    def setup_test_input(self, sess, class_indices, labels_table, num_samples, filenames):

        if num_samples <= self.num_val_trials:
            dataset_chunk = num_samples
        else:
            dataset_chunk = min(num_samples, max(2 * len(class_indices), self.num_val_trials))

        iterator, image_batch, label_batch, _ = dat.deploy_dataset(filenames, labels_table, dataset_chunk,
                                                                   self.image_dims, shuffle=False)
        labels_table.init.run(session=sess)
        sess.run(iterator.initializer)
        ims, labs = sess.run([image_batch, label_batch])
        inputs, targets_one_hot, labels_left, labels_right = self._make_kshot_task(self.num_val_trials, ims,
                                                                                     labs, self.num_val_ways)
        targets = np.argmax(targets_one_hot, axis=1)
        return inputs, targets, targets_one_hot, labels_left, labels_right

    def setup_network(self, num_classes):
        if self.optimizer == 'sgd':
            self.optimizer = Modified_SGD(
                lr=self.learning_rate,
                momentum=0.5)
        elif self.optimizer == 'adam':
            self.optimizer = Adam(self.learning_rate)
        else:
            raise ("optimizer not known")

        model_class = util.str_to_class(self.model_name)
        siamese_network = model_class(self.image_dims, self.optimizer)
        self.embeddings, self.model = siamese_network.build_net()
        with open(os.path.join(self.results_path, 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
        if self.checkpoint:
            self.model.load_weights(os.path.join(self.checkpoint, "weights.h5"))

    def train_triplet(self, dat_info):
        self.num_train_cls = len(dat_info.train_class_indices)
        self.setup_network(self.num_train_cls)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        train_table = dat.make_hashtable(dat_info.train_class_indices, list(range(len(dat_info.train_class_indices))))

        train_iterator, train_image_batch, train_label_batch, _ = dat.deploy_dataset(
            dat_info.train_filenames, train_table, self.batch_size, self.image_dims, shuffle=True)

        val_inputs, val_targets, val_targets_one_hot, \
        val_labels_left, val_labels_right = self.setup_test_input(sess, dat_info.val_class_indices, train_table,
                                                                  dat_info.num_val_samples, dat_info.val_filenames)

        test_table = dat.make_hashtable(dat_info.test_class_indices, list(range(len(dat_info.test_class_indices))))
        test_inputs, test_targets, test_targets_one_hot, \
        test_labels_left, test_labels_right = self.setup_test_input(sess, dat_info.test_class_indices, test_table,
                                                                    dat_info.num_test_samples, dat_info.test_filenames)

        predictions_train = self.model(train_image_batch)

        if self.triplet_strategy == "batch_all":
            loss, fraction = batch_all_triplet_loss(train_label_batch, predictions_train, margin=0.5,
                                                    squared=False)
        elif self.triplet_strategy == "batch_hard":
            loss = batch_hard_triplet_loss(train_label_batch, predictions_train, margin=0.5,
                                           squared=False)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(self.triplet_strategy))

        # Summaries for training
        tf.summary.scalar('loss', loss)
        if self.triplet_strategy == "batch_all":
            tf.summary.scalar('fraction_positive_triplets', fraction)
        tf.summary.image('train_image', train_image_batch, max_outputs=1)

        updates_op = self.optimizer.get_updates(
            params=self.model.trainable_weights,
            loss=loss)

        train = K.function(
            inputs=[train_image_batch, train_label_batch],
            outputs=[loss],
            updates=updates_op)

        self.validate(0, 0, val_inputs, val_targets, val_targets_one_hot,
                      dat_info.val_class_names, val_labels_left, val_labels_right, test_inputs,
                      test_targets, test_targets_one_hot, dat_info.test_class_names,
                      test_labels_left, test_labels_right)

        for epoch in range(self.num_epochs):
            sess.run(train_iterator.initializer)
            batch_index = 0
            logger.info("Training epoch {} ...".format(epoch))
            losses_train = []
            while True:
                try:
                    train_ims, train_labs = sess.run([train_image_batch, train_label_batch])
                    # deals with unequal tf record lengths, when the batch might have samples
                    # from only one class
                    if len(np.unique(train_labs)) < 2 or \
                            len(np.unique(train_labs)) > self.batch_size // 2:
                        continue
                    loss_train = train([train_ims, train_labs])
                    losses_train.append(loss_train[0])
                    batch_index += 1
                except tf.errors.OutOfRangeError:
                    print(np.mean(losses_train))
                    if (epoch+1) % self.evaluate_every == 0:
                        self.validate(epoch, batch_index, val_inputs, val_targets, val_targets_one_hot,
                                      dat_info.val_class_names, val_labels_left, val_labels_right, test_inputs,
                                      test_targets, test_targets_one_hot, dat_info.test_class_names,
                                      test_labels_left, test_labels_right)
                    break

    def validate(self, epoch, batch_index, val_inputs, val_targets, val_targets_one_hot,
                 val_class_names, val_labels_left, val_labels_right, test_inputs, test_targets,
                 test_targets_one_hot, test_class_names, test_labels_left, test_labels_right):

        epoch_folder = os.path.join(self.results_path, "epoch_{}".format(epoch))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)

        eval_val = self.eval("validation", epoch, val_inputs, val_targets, val_targets_one_hot, val_class_names,
                             val_labels_left, val_labels_right)
        logger.info("Siamese {} way {}-shot accuracy on known classes: {}% on classes {}"
                    "".format(self.num_val_ways, self.num_shots, eval_val.siam_accuracy * 100, val_class_names))

        eval_test = self.eval("test", epoch, test_inputs, test_targets, test_targets_one_hot, test_class_names,
                              test_labels_left, test_labels_right)
        logger.info("Siamese {} way {}-shot accuracy on novel classes: {}% on classes {}"
                    "".format(self.num_val_ways, self.num_shots, eval_test.siam_accuracy * 100, test_class_names))

        util.metrics_to_csv(os.path.join(epoch_folder, "metrics_epoch_{}.csv".format(epoch)),
                            np.asarray([eval_val.siam_accuracy, eval_test.siam_accuracy]),
                            ["siamese_val_accuracy", "siamese_test_accuracy"])

        if self.save_weights:
            # self.net.save_weights(os.path.join(self.results_path, "weights.h5"))
            self.model.save(os.path.join(self.results_path, "weights.h5"), overwrite=True, include_optimizer=False)

        if self.write_to_tensorboard:
            self._write_logs_to_tensorboard(batch_index, eval_val.siam_accuracy, eval_test.siam_accuracy)


    def test(self,dat_info):
        num_train_cls = len(dat_info.train_class_indices)
        self.setup_network(num_train_cls)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        test_table = dat.make_hashtable(dat_info.test_class_indices, list(range(len(dat_info.test_class_indices))))
        test_inputs, test_targets, test_targets_one_hot, \
        test_labels_left, test_labels_right = self.setup_test_input(sess, dat_info.test_class_indices, test_table,
                                                                    dat_info.num_test_samples, dat_info.test_filenames)

        eval_results = self.eval(test_inputs, test_targets, test_targets_one_hot, dat_info.test_class_names, test_labels_left,
                                 test_labels_right)

        util.metrics_to_csv(os.path.join(self.results_path, "inference.csv"), np.asarray([eval_results.siam_accuracy]),
                            ["test_acc"])

    def eval(self, mode, epoch, inps, targets, targets_one_hot, class_names, eval_labels_left, eval_labels_right):
        logger.info(
            "Evaluating model on {} random {} way one-shot learning tasks from classes {}"
            "...".format(self.num_val_trials, self.num_val_ways, class_names))

        siamese_preds = np.zeros((self.num_val_trials, self.num_shots))

        for trial in range(self.num_val_trials):
            for s in range(self.num_shots):
                embeddings_left = self.model.predict(inps[0][trial, :, s])
                embeddings_right = self.model.predict(inps[1][trial, :, s])
                import scipy
                # if trial < 5:
                #     vis.plot_validation_images(
                #         "/media/iulialexandra/Storage/code/low-shot/siamese_on_edge/results/val_images_shot{}.png".format(s),
                #                                [inps[0][trial, :, s], inps[1][trial, :, s]], targets_one_hot[trial, :, s])
                dists = []
                for way in range(self.num_val_ways):
                    dists.append(scipy.spatial.distance.euclidean(embeddings_left[way], embeddings_right[way]))
                siamese_preds[trial, s] = np.argmin(dists)
        interm_siam_acc = np.equal(siamese_preds, targets)
        siam_accuracy = np.mean(interm_siam_acc)
        EvalResults = namedtuple("EvalResults", ["siam_accuracy"])
        self.plot_emb_tsne(inps, eval_labels_left, mode, epoch)
        return EvalResults(siam_accuracy)

    def plot_emb_tsne(self, inputs, labels, mode, epoch):
        num_trials = 100
        tsne = TSNE()
        embeddings = []
        labs = np.ravel(labels[:num_trials])
        for trial in range(num_trials):
            embeddings.append(self.model.predict(np.squeeze(inputs[0][trial])))
        embs = np.vstack(embeddings)
        tsne_embeds = tsne.fit_transform(embs)
        vis.scatter(tsne_embeds, labs, mode, epoch, self.results_path)

    def _get_train_balanced_batch(self, images, labels, num_train_cls):
        with tf.device('/cpu:0'):
            labels = np.ravel(labels)
            targets = np.zeros((self.batch_size,), dtype=np.int8)
            shuffle_indices = rng.choice(range(self.batch_size), size=self.batch_size // 2)
            targets[shuffle_indices] = 1
            image_pairs = np.zeros((self.batch_size, 2,) + np.shape(images)[1:], dtype=np.float32)
            # choose the first row of images
            chosen_indices = rng.choice(range(len(labels)), size=self.batch_size, replace=False)
            left_labels = labels[chosen_indices]
            right_labels = []
            image_pairs[:, 0, ...] = images[chosen_indices]
            for i, index in enumerate(chosen_indices):
                comparative_lbl = labels[index]
                if targets[i] == 0:
                    pair_index = rng.choice(np.where(labels != comparative_lbl)[0], size=(1,))[0]
                    image_pairs[i, 1, ...] = images[pair_index]
                    right_labels.append(labels[pair_index])
                elif targets[i] == 1:
                    pair_index = rng.choice(np.where(labels == comparative_lbl)[0], size=(1,))[0]
                    image_pairs[i, 1, ...] = images[pair_index]
                    right_labels.append(labels[pair_index])
            right_labels = np.array(right_labels)
            right_labels = to_categorical(right_labels, num_classes=num_train_cls)
            left_labels = to_categorical(left_labels, num_classes=num_train_cls)

            return (image_pairs[:, 0, ...], image_pairs[:, 1, ...],
                    left_labels, np.expand_dims(targets, 1), right_labels)

    def _make_kshot_task(self, n_val_tasks, image_data, labels, n_ways):
        with tf.device('/cpu:0'):
            classes = np.unique(labels)
            new_labels =  np.array([np.where(classes == lab)[0][0] for lab in labels])
            assert len(classes) == n_ways
            if len(image_data) < n_val_tasks:
                replace = True
            else:
                replace = False
            reference_indices = rng.choice(range(len(labels)), size=(n_val_tasks,), replace=replace)
            reference_new_labels = new_labels[reference_indices]
            reference_labels = labels[reference_indices]
            comparison_indices = np.zeros((n_val_tasks, n_ways, self.num_shots), dtype=np.int32)
            targets = np.zeros((n_val_tasks, n_ways, self.num_shots))
            targets[range(n_val_tasks), reference_new_labels, :] = 1
            for i, cls in enumerate(classes):
                cls_indices = np.where(labels == cls)[0]
                comparison_indices[:, i] = rng.choice(cls_indices, size=(n_val_tasks, self.num_shots),
                                                      replace=True)
            comparison_images = image_data[comparison_indices, ...]
            reference_images = image_data[reference_indices, np.newaxis, np.newaxis, ...]
            reference_images = np.repeat(reference_images, n_ways, axis=1)
            reference_images = np.repeat(reference_images, self.num_shots, axis=2)
            image_pairs = [np.array(reference_images, dtype=np.float32),
                           np.array(comparison_images, dtype=np.float32)]

            labels_left = np.repeat(np.reshape(reference_labels, (1000, 1, 1)), n_ways, axis=1)
            labels_left = np.repeat(labels_left, self.num_shots, axis=2)
            labels_right = labels[comparison_indices]

            return image_pairs, targets, labels_left, labels_right

    def _write_logs_to_tensorboard(self, batch_index, left_loss, left_acc, right_loss, right_acc,
                                   siamese_loss, siamese_acc, val_accuracy, test_accuracy,
                                   test_probs_std, test_probs_means):
        """ Writes the logs to a tensorflow log file
        """

        summary = tf.Summary()

        value = summary.value.add()
        value.simple_value = left_loss
        value.tag = 'Left images classification training loss'

        value = summary.value.add()
        value.simple_value = left_acc
        value.tag = 'Left images classification training accuracy'

        value = summary.value.add()
        value.simple_value = right_loss
        value.tag = 'Right images classification training loss'

        value = summary.value.add()
        value.simple_value = right_acc
        value.tag = 'Right images classification training accuracy'

        value = summary.value.add()
        value.simple_value = siamese_loss
        value.tag = 'Siamese training loss'

        value = summary.value.add()
        value.simple_value = siamese_acc
        value.tag = 'Siamese training accuracy'

        value = summary.value.add()
        value.simple_value = val_accuracy
        value.tag = 'Siamese validation accuracy'

        value = summary.value.add()
        value.simple_value = test_accuracy
        value.tag = 'Siamese testing accuracy on unseen classes'

        means_hist_summary = self._log_tensorboard_hist(test_probs_means,
                                                        'Siamese prediction probabilities means')
        std_hist_summary = self._log_tensorboard_hist(test_probs_std,
                                                      'Siamese prediction probabilities'
                                                      ' standard deviation')

        self.summary_writer.add_summary(summary, batch_index)
        self.summary_writer.add_summary(means_hist_summary, batch_index)
        self.summary_writer.add_summary(std_hist_summary, batch_index)
        self.summary_writer.flush()

    def _log_tensorboard_hist(self, values, tag, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        return summary
