from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Input, Dropout, \
    concatenate, Lambda
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tools.modified_sgd import Modified_SGD


class TripletV1:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer

    def build_net(self):

        net_input = Input(shape=self.input_shape, name="network_input")

        block_name = "block1"
        conv0 = Conv2D(32, (9, 9), padding="same", activation='relu',
                       kernel_initializer="he_normal", name='{}_conv0'.format(block_name),
                       kernel_regularizer=l2(1e-2))(net_input)
        conv0 = BatchNormalization(name='{}_conv0_batchnorm'.format(block_name))(conv0)

        conv1 = Conv2D(64, (7, 7), padding="same", activation='relu',
                       kernel_initializer="he_normal", name='{}_conv1'.format(block_name),
                       kernel_regularizer=l2(1e-2))(conv0)
        conv1 = BatchNormalization(name='{}_conv1_batchnorm'.format(block_name))(conv1)

        conv2 = Conv2D(128, (5, 5), padding="same", activation='relu',
                       kernel_initializer="he_normal", name='{}_conv2'.format(block_name),
                       kernel_regularizer=l2(1e-2))(conv1)
        conv2 = BatchNormalization(name='{}_conv2_batchnorm'.format(block_name))(conv2)

        conv3 = Conv2D(256, (3, 3), padding="same", activation='relu',
                       kernel_initializer="he_normal", name='{}_conv3'.format(block_name),
                       kernel_regularizer=l2(1e-2))(conv2)
        conv3 = BatchNormalization(name='{}_conv3_batchnorm'.format(block_name))(conv3)
        conv3 = MaxPooling2D(padding="valid")(conv3)
        conv4 = Conv2D(128, (3, 3), padding="same", activation='relu',
                       kernel_initializer="he_normal", name='{}_conv4'.format(block_name),
                       kernel_regularizer=l2(1e-2))(conv3)
        conv4 = BatchNormalization(name='{}_conv4_batchnorm'.format(block_name))(conv4)
        net = MaxPooling2D(padding="valid")(conv4)
        net = Flatten()(net)
        embedding = Dense(64, activation=None, kernel_regularizer=l2(1e-2),
              kernel_initializer="he_normal", name='fc1')(net)

        net_output = Model(inputs=net_input, outputs=embedding)
        return embedding, net_output


def TripletV2(is_training, images, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.shape[1:] == [7, 7, num_channels * 2]

    out = tf.reshape(out, [-1, 7 * 7 * num_channels * 2])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out