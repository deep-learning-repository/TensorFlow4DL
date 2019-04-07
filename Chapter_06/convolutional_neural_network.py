import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets('./DATA', one_hot=True)

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
VALIDATION_SIZE = 5000
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 128
EVAL_BATCH_SIZE = BATCH_SIZE
KEEP_PROP = 0.5
LEARNING_RATE = 0.001
EPOCHS = 20
DISPLAY_STEP = 5

train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

# Generate a validation set.
validation_data = test_data[:VALIDATION_SIZE]
validation_labels = test_labels[:VALIDATION_SIZE]
test_data = test_data[VALIDATION_SIZE:]
test_labels = test_labels[VALIDATION_SIZE:]


def reshape_data(data):
    """ Reshaped values to [num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]
    """
    data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


test_data = reshape_data(test_data)
validation_data = reshape_data(validation_data)

conv_1_weights = tf.Variable(
    tf.truncated_normal([5, 5, 1, 32],  # 5x5 filter, depth 32.
                        stddev=0.1,
                        seed=0,
                        dtype=tf.float32))
conv_1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))

conv_2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64],
                                                 stddev=0.1,
                                                 seed=0,
                                                 dtype=tf.float32))
conv_2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))

fc_1_weights = tf.Variable(
    tf.truncated_normal([28 // 4 * 28 // 4 * 64, 120],
                        stddev=0.1,
                        seed=0,
                        dtype=tf.float32))
fc_1_biases = tf.Variable(tf.zeros(shape=[120], dtype=tf.float32))

fc_2_weights = tf.Variable(tf.truncated_normal([120, 84],
                                               stddev=0.1,
                                               seed=0,
                                               dtype=tf.float32))
fc_2_biases = tf.Variable(tf.zeros(shape=[84], dtype=tf.float32))

logit_weights = tf.Variable(tf.truncated_normal([84, NUM_LABELS],
                                                stddev=0.1,
                                                seed=0,
                                                dtype=tf.float32))
logit_biases = tf.Variable(tf.zeros(shape=[NUM_LABELS], dtype=tf.float32))

data_node = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1))
label_node = tf.placeholder(tf.int64, shape=(None, NUM_LABELS))
keep_prob = tf.placeholder(tf.float32)


def model(data):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv_1 = tf.nn.conv2d(data,
                          conv_1_weights,
                          strides=[1, 1, 1, 1],
                          padding='SAME')

    # Bias and rectified linear non-linearity.
    conv_1 = tf.nn.relu(conv_1 + conv_1_biases)
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    conv_2 = tf.nn.conv2d(pool_1,
                          conv_2_weights,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    conv_2 = tf.nn.relu(conv_2 + conv_2_biases)
    pool_2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    flatten_layer = tf.contrib.layers.flatten(pool_2)
    # fully connected layers.
    fully_connected_1 = tf.nn.relu(tf.matmul(flatten_layer, fc_1_weights) + fc_1_biases)

    fully_connected_2 = tf.nn.relu(tf.matmul(fully_connected_1, fc_2_weights) + fc_2_biases)

    fully_connected_2 = tf.nn.dropout(fully_connected_2, keep_prob=keep_prob)
    logits = tf.matmul(fully_connected_2, logit_weights) + logit_biases
    return logits


def get_accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    return accuracy


def get_batch_predictions(data, keep_prob_, sess):
    """Get predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size is larger than dataset: %d" % size)
    predictions_lst = np.ndarray(shape=(size, NUM_LABELS),
                                 dtype=np.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions_lst[begin:end, :] = sess.run(predictions, feed_dict={data_node: data[begin:end, ...],
                                                                             keep_prob: keep_prob_})
        else:
            batch_predictions = sess.run(predictions, feed_dict={data_node: data[-EVAL_BATCH_SIZE:, ...],
                                                                 keep_prob: keep_prob_})
            predictions_lst[begin:, :] = batch_predictions[begin - size:, :]
    return predictions_lst


def get_prediction_accuracy(predictions, labels):
    """Return the based based on dense predictions and sparse labels."""
    return np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


logits = model(data=data_node)
predictions = tf.nn.softmax(logits)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_node))
optimization = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

accuracy = get_accuracy(logits, label_node)

# Create a local session to run the training.
init_g = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_g)
    for epoch in range(EPOCHS):
        batch_X, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
        batch_X = reshape_data(batch_X)
        _ = sess.run([optimization], feed_dict={data_node: batch_X, label_node: batch_y, keep_prob: KEEP_PROP})
        if (epoch + 1) % DISPLAY_STEP == 0:
            l, acc = sess.run([loss, accuracy], feed_dict={data_node: batch_X, label_node: batch_y, keep_prob: KEEP_PROP})
            val_predictions = get_batch_predictions(data=validation_data, keep_prob_=1, sess=sess)
            val_acc = get_prediction_accuracy(val_predictions, validation_labels)
            print(f'Epoch: {epoch + 1}, mini_batch_loss= {round(float(l), 2)}, training accuracy: {round(float(acc), 2)},'
                  f' validation accuracy: {round(float(val_acc), 2)}')
    print('Optimization Finished!!')
    test_predictions = get_batch_predictions(data=test_data, keep_prob_=1, sess=sess)
    print(f'Testing Accuracy: {get_prediction_accuracy(test_predictions,test_labels)}')
    saver.save(sess, 'CheckPoint/convolutional_neural_network.chkp')

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'CheckPoint/convolutional_neural_network.chkp')
n_image = 25
images = np.array(random.choices(mnist.test.images, k=n_image)).reshape(-1, 28, 28, 1)
predictions = sess.run(tf.argmax(logits, axis=1), feed_dict={data_node: images, keep_prob: 1})
for i in range(n_image):
    plt.imshow(images[i].reshape((28, 28)))
    plt.show()
    print(f'class prediction is {predictions[i]}')
