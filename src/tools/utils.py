import os
import sys
import logging
import json
import datetime
import csv
import numpy as np
import signal
from random import shuffle
from pathlib import Path
from networks.original_nets import *
from networks.horizontal_nets import *
from networks.resnets import *
import random as rn

import time
import logging

logger = logging.getLogger()


class Uninterrupt(object):
    """From https://gist.github.com/nonZero/2907502
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """
    def __init__(self, sigs=[signal.SIGINT], verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.verbose:
                print("Interruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
            self.orig_handlers = None


def str_to_class(str):
    """Gets a class when given its name.

    Args:
        str: the name of the class to retrieve
    """
    return getattr(sys.modules[__name__], str)


def make_results_dir(args):
    """Makes one folder for the results using the current date and time, another that holds
    the most recent experiment data and initializes the logger.
    """
    now = datetime.datetime.now()
    date = "{}_{}_{}-{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                         now.second, now.microsecond)
    if args.results_path is None:
        parent_path = Path().absolute().parent
        save_path = os.path.join(parent_path, "results")
    else:
        save_path = args.results_path
    if args.left_classif_factor > 0:
        args.with_classif = "yes"
    else:
        args.with_classif = "no"
    results_path = os.path.join(save_path, str(date) + "_seed_" + str(args.seed) + "_"
                                + args.dataset + "_" + args.model + "_" + args.with_classif)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = initialize_logger(results_path, args.console_print)
    return results_path, logger


def experiment_details(args):
    """Writes the arguments to file to keep track of various experiments"""
    exp_path = os.path.join(args.results_path, "experiment_details.json")
    if not os.path.exists(exp_path):
        with open(exp_path, "w") as outfile:
            json.dump(vars(args), outfile)


def initialize_logger(output_dir, print_to_console):
    """initializes loggers for debug and error, prints to file

    Args:
        output_dir: the directory of the logger file
    """

    logger = logging.getLogger("siam_logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = 0
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  "%d-%m-%Y %H:%M:%S")
    # Setup file logging
    fh = logging.FileHandler(os.path.join(output_dir, "log.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Setup console logging
    if print_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def initialize_experiment(args, train=True):
    if train:
        args.results_path, logger = make_results_dir(args)
        experiment_details(args)
    else:
        args.results_path = args.chkpt
        logger = initialize_logger(args.results_path, True)

    # make the experiment deterministic
    np.random.seed(args.seed)
    rn.seed(args.seed)
    tf.set_random_seed(args.seed)
    os.environ["PYTHONSEED"] = str(args.seed)
    return args, logger


def sweep_directories(directory_list, token=None):
    """Given a list of directories, it returns the files inside"""
    files = []
    if type(directory_list) != list:
        directory_list = [os.path.join(directory_list, direc) for direc in
                          os.listdir(directory_list)]
    for directory in directory_list:
        for f in os.listdir(directory):
            file = os.path.join(directory, f)
            if (os.path.isfile(file) and token in file):
                files.append(file)
    shuffle(files)
    return files


def timed(method):
    def timeit(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        elapsed_time = end_time - start_time
        logger.info("{} run completed in {:.3f}s".format(method.__name__, elapsed_time))
        return result

    return timeit


def metrics_to_csv(filepath, values, fieldnames):
    with open(filepath, 'w') as csvfile:
        values_dict = dict(zip(fieldnames, values))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(values_dict)


def nearest_neighbour_correct(pairs, targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i] ** 2 - pairs[1][i] ** 2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


def all_diffs(a, b):
    # Returns a tensor of all combinations of a - b
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def euclidean_dist(embed1, embed2):
    # Measures the euclidean dist between all samples in embed1 and embed2

    diffs = all_diffs(embed1, embed2)  # get a square matrix of all diffs
    return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)


def bh_triplet_loss(dists, labels, margin):
    # Defines the "batch hard" triplet loss function.

    same_identity_mask = tf.equal(tf.expand_dims(labels, axis=1),
                                  tf.expand_dims(labels, axis=0))
    negative_mask = tf.logical_not(same_identity_mask)
    positive_mask = tf.logical_xor(same_identity_mask,
                                   tf.eye(tf.shape(labels)[0], dtype=tf.bool))

    furthest_positive = tf.reduce_max(dists * tf.cast(positive_mask, tf.float32), axis=1)
    closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                 (dists, negative_mask), tf.float32)

    diff = furthest_positive - closest_negative

    return tf.maximum(diff + margin, 0.0)
