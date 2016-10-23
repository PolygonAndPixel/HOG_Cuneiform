""" Getting started with autoencoder and tensorflow. MNIST is used as data.
Based on https://github.com/pkmital/tensorflow_tutorials/blob/master/python/07_autoencoder.py

Maicon Hieronymus, August 2016
"""
# Used for MNIST test
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc

import numpy as np
import math
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from random import randint
from os import path
from sklearn import svm


plot = False
gaussian = False
filename = "Autoencoder"
learning_rate = 0.001
batch_size = 50
n_epochs = 10
verbosity = 0
no_of_classes = 12
extension = 1000
orig = False
confidence = 0.9999
adam = 1e-4

# With help from http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def sliding_window(image, step_size, window_size):
    """Slide over an image with a quadratic window and yield the current part
    of the image.

    """
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size, x:x + window_size])


def create_rectangle(image, x, y, label, width, height):
    """Create a rectangle with given width and height. The coordinates are the
    upper left corner of the rectangle. The label indicates a certain color.
    """
    colorrange = 256*256*256
    # Plus 2 to avoid black and white
    my_color = colorrange/(no_of_classes+2)
    my_color = my_color*(label+1)
    rgb = [0, 0, 0]
    if my_color >= 256*256:
        rgb[0] = my_color%256
        rgb[1] = my_color%(256*256) / 256
        rgb[2] = my_color/(256*256)
    elif my_color > 255:
        rgb[0] = my_color%256
        rgb[1] = my_color/256
    else:
        rgb[2] = my_color

    for i in range(0,height):
        for j in range(0,width):
            if(i == 0 or i == height-1):
                image[y+i][x+j][0] = rgb[0]
                image[y+i][x+j][1] = rgb[1]
                image[y+i][x+j][2] = rgb[2]
            if(j == 0 or j == width-1):
                image[y+i][x+j][0] = rgb[0]
                image[y+i][x+j][1] = rgb[1]
                image[y+i][x+j][2] = rgb[2]


def autoencoder(dimensions=[16384, 512, 256, 64]):
    """Build a deep autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.


    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # Input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # Latent representation
    z = current_input
    encoder.reverse()

    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # Now have the reconstruction through the network
    y = current_input

    # Cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


def create_legend(image, y_coord, examples, labels, image_width, image_height,
        sklearn):
    """ Create a legend at the bottom beginning at y_coord.
    """
    symbol_per_row = len(image[0])/30
    current_column = 0
    current_row = 0
    if(verbosity > 0):
        print "Creating legend"

    for i in range(0, no_of_classes):
        if(verbosity > 0):
            print "Looking for example for label ", i
        label = -1
        idx = 0
        if(sklearn):
            while(label != i):
                label = labels[idx]
                idx = idx+1
        else:
            while(label != i):
                label = np.argmax(labels[idx])
                idx = idx+1
        idx = idx-1
        y_start = y_coord + 1 + current_row*30

        x_start = current_column * 30
        for y in range(0, image_height):
            for x in range(0, image_width):
                # print image[y_coord + 1 + i*2 + i*28 + y][x][0]
                # print np.shape(examples)
                # print examples[idx][y][x]*255
                tmp = np.absolute(examples[idx][y*28+x]-1)
                image[y_start + y][x_start+x][0] = tmp*255
                image[y_start + y][x_start+x][1] = tmp*255
                image[y_start + y][x_start+x][2] = tmp*255
        create_rectangle(image, x_start,
                y_start, i,
                image_width, image_height)
        current_column = current_column+1
        if(current_column == symbol_per_row):
            current_column = 0
            current_row = current_row + 1


def load_images():
    images = []
    label = []
    if(orig):
        fna = "labeled_data_orig/"
    else:
        fna = "labeled_data/"
    for j in range(1, no_of_classes+1):
        i = 1
        fname = fna + str(j) + "/" + str(i) + ".png"
        while(path.isfile(fname)):
        # for i in range(1,entries_per_class+1):
        #     fname = fna + str(j) + "/" + str(i) + ".png"
            read = misc.imread(fname)
            if(np.shape(read) == (128,128,3)):
                data = read[:,:,0]
            else:
                data = read
            data = np.reshape(data, len(data[0])*len(data[1]))
            data = data.astype('f')

            data = data - 255
            data = np.absolute(data)
            data = data/255

            tmp_label = np.zeros(no_of_classes, dtype='f')
            tmp_label[j-1] = 1.
            images.append(data)
            label.append(tmp_label)
            i = i+1
            fname = fna + str(j) + "/" + str(i) + ".png"
    return images, label


def extend_set(data, labels, extension, sklearn):
    """Extend the dataset by copying randomly the entries. If gaussian has been
    set to true, gaussian noise will be added. The images will be rotated
    randomly by -15 to +15 degrees.

    Parameters
    ----------


    Returns
    -------
    new_data :   List
                 List of flattened images.
    new_labels : List
                 List of labels. If sklearn is true, then the labels are values
                 from 0 to number of labels. Else it is a list of length
                 number of labels with 1 for its label and 0 else.

    """
    new_length = extension*len(data)
    new_data = []
    new_labels = []
    for i in range(0,new_length):
        rotate = randint(-360,360)
        idx = randint(0,len(data)-1)
        # idx = randint(0,entries_per_class*no_of_classes-1)
        if(rotate < 15 and rotate > -15):
            tmp_data = np.reshape(data[idx], (28,28))
            # print data[idx]
            tmp_data = misc.imrotate(tmp_data, rotate)
            tmp_data = np.reshape(tmp_data, 28*28)
            tmp_data = tmp_data.astype('f')
            # print tmp_data
            # tmp_data = tmp_data - 255
            # print tmp_data
            # tmp_data = np.absolute(tmp_data)
            tmp_data = tmp_data/255

            # print tmp_data
            # exit()

        else:
            tmp_data = data[idx]
        # Adding some Gaussian noise
        if(gaussian):
            tmp_data = tmp_data + np.random.normal(0, 0.05, len(tmp_data))
            tmp_data = np.absolute(tmp_data)

        new_data.append(tmp_data)
        if(sklearn):
            new_labels.append(np.argmax(labels[idx]))
        else:
            new_labels.append(labels[idx])
    return new_data, new_labels


# Basic test with MNIST
def test_mnist():
    """Test the autoencoder using MNIST."""
    image_width = 128
    image_size = image_width*image_width
    #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    cuneiform = load_images()
    mean_img = np.mean(cuneiform, axis=0)
    ae = autoencoder(dimensions=[image_size, 256, 64])

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Fit the training data
    if(verbosity > 0):
        for epoch_i in range(n_epochs):
            for batch_i in range(len(cuneiform) // batch_size):
                batch_xs = cuneiform[batch_i*batch_size:(batch_i+1)*batch_size]
                # batch_xs, _ = mnist.train.next_batch(batch_size)
                train = np.array([img - mean_img for img in batch_xs])
                sess.run(optimizer, feed_dict={ae['x']: train})
            if epoch_i%100 == 0:
                print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))
    else:
        for epoch_i in range(n_epochs):
            for batch_i in range(len(cuneiform) // batch_size):
                batch_xs, = cuneiform[batch_i*batch_size:(batch_i+1)*batch_size]
                #batch_xs, _ = mnist.train.next_batch(batch_size)
                train = np.array([img - mean_img for img in batch_xs])
                sess.run(optimizer, feed_dict={ae['x']: train})
    if(plot):
        # Plot example reconstructions
        n_examples = 20
        test_xs = np.array(cuneiform[0:n_examples])
        #test_xs, _ = mnist.test.next_batch(n_examples)
        test_xs_norm = np.array([img - mean_img for img in test_xs])
        recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
        fig, axs = plt.subplots(2, n_examples, figsize=(40, 8))
        for example_i in range(n_examples):
            # for i in range(example_i, n_examples):
            #     np.reshape(test_xs[i], (image_width, image_width))
            # axs[0][example_i].imshow(test_xs[example_i, :])
                # np.reshape(test_xs[example_i, :], (image_width, image_width)))
            axs[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (image_width, image_width)))
            axs[1][example_i].imshow(
                np.reshape([recon[example_i, :] + mean_img], (image_width, image_width)))
        plt.show()
        #plt.savefig(filename, dpi=300)


def test_mnist2():
    """Test another autoencoder using MNIST.
    Code is from https://github.com/nlintz/TensorFlow-Tutorials/blob/master/06_autoencoder.py"""
    mnist_width = 28
    n_visible = mnist_width * mnist_width
    n_hidden = 500
    corruption_level = 0.3

    # create node for input data
    X = tf.placeholder("float", [None, n_visible], name='X')

    # create node for corruption mask
    mask = tf.placeholder("float", [None, n_visible], name='mask')

    # create nodes for hidden variables
    W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
    W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                            minval=-W_init_max,
                            maxval=W_init_max)

    W = tf.Variable(W_init, name='W')
    b = tf.Variable(tf.zeros([n_hidden]), name='b')

    W_prime = tf.transpose(W)  # tied weights between encoder and decoder
    b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')


    def model(X, mask, W, b, W_prime, b_prime):
        tilde_X = mask * X  # corrupted X

        Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state
        Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input
        return Z

    # build model graph
    Z = model(X, mask, W, b, W_prime, b_prime)

    # create cost function
    cost = tf.reduce_sum(tf.pow(X - Z, 2))  # minimize squared error
    train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

    # load MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()
        if(plot):
            for i in range(n_epochs):
                for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                    input_ = trX[start:end]
                    mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
                    sess.run(train_op, feed_dict={X: input_, mask: mask_np})

                mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
                print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))
        else:
            for i in range(n_epochs):
                for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                    input_ = trX[start:end]
                    mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
                    sess.run(train_op, feed_dict={X: input_, mask: mask_np})

                mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)

        #if(plot):
            ## Plot example reconstructions
            #mean_img = np.mean(mnist.train.images, axis=0)
            #n_examples = 15
            #test_xs, _ = mnist.test.next_batch(n_examples)
            #test_xs_norm = np.array([img - mean_img for img in test_xs])
            #recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
            #fig, axs = plt.subplots(2, n_examples, figsize=(40, 8))
            #for example_i in range(n_examples):
                #axs[0][example_i].imshow(
                    #np.reshape(test_xs[example_i, :], (28, 28)))
                #axs[1][example_i].imshow(
                    #np.reshape([recon[example_i, :] + mean_img], (28, 28)))
            #plt.savefig(filename, dpi=300)


def classify(fname, sess, image_width, x, y, cuneiform, cuneiform_label):
    """Use fname as input to identify the letters. Uses sliding window and saves
    an image with found letters in a box.

    Parameters
    ----------
    fname :         Path
                    Input with cuneiform script as jpg. (png should work too)
    sess :          Tensorflow session
                    The trained session which classifies the image.
    image_width :   Integer
                    Width of window used in sliding window. Windows are squared.
    x :             Tensorflow placeholder
                    Should be something like
                    tf.placeholder(tf.float32, [None, image_size]).
                    Input layer for your network.
    y :             Tensorflow activation
                    Should be something like tf.nn.softmax(...).
                    Output layer for your network.
    cuneiform :     List with shape (no_of_images, image_size)
                    The set which has been used to train the network.
    cuneiform_label :   List with shape (no_of_images, no_of_classes)
                        List with labels where each label is a list. The label
                        corresponds to the index with entry 1. Else it is 0.
    """
    image = misc.imread(fname)
    data = []
    coord = {'x': [], 'y': []}
    for(x_coord, y_coord, window) in sliding_window(image, 4, image_width):
        # Ignore windows with different sizes at en edge.
        if window.shape[0] != image_width or window.shape[1] != image_width:
            continue
        # Add the window to our dataset
        # window = window[:,:,0]
        window = np.reshape(window, image_width*image_width)
        window = window.astype('f')
        window = window - 255
        window = np.absolute(window)
        window = window/255

        data.append(window)
        coord['x'].append(x_coord)
        coord['y'].append(y_coord)

    predictions = {'x': [], 'y': [], 'label': [], 'confidence': []}
    max_confidence = 0.0
    global confidence
    for i in range(0,len(data)):
        classification = sess.run(y, feed_dict={x: [data[i]]})
        label = classification[0].argmax()
        if(classification[0][label] > confidence and label != 2):
            predictions['x'].append(coord['x'][i])
            predictions['y'].append(coord['y'][i])
            predictions['label'].append(label)
            predictions['confidence'].append(classification[0][label])

            ####################################################################
            # imName = 'windows/' + str(label+1) + '_' + str(i) + '.png'
            # img = np.zeros((28,28), dtype = int)
            # for y_c in range(0, image_width):
            #     for x_c in range(0, image_width):
            #         tmp = np.absolute(data[i][y_c*28+x_c]-1)
            #         img[y_c][x_c] = tmp*255
            # misc.imsave(imName, img)

        if(max_confidence < classification[0][label]):
            max_confidence = classification[0][label]
        # if(plot):
        #     plt.imshow(data[i].reshape(image_width, image_width),
        #             cmap=plt.cm.binary)
        #     plt.show()
    if(verbosity > 0):
        print "Number of found signs: ", len(predictions['x'])
        print "Best confidence: ", max_confidence
    ### FOR DEBUG ONLY!
    percentage = confidence
    if(max_confidence > confidence):
        max_confidence = confidence
    confidence = max_confidence
    while ( len(predictions['x']) < 30 and percentage > 0):
        #predictions.clear()
        predictions = {'x': [], 'y': [], 'label': [], 'confidence': []}

        print "Trying another confidence with ", confidence
        for i in range(0,len(data)):
            classification = sess.run(y, feed_dict={x: [data[i]]})
            label = classification[0].argmax()
            if(classification[0][label] > confidence and label != 2):
                predictions['x'].append(coord['x'][i])
                predictions['y'].append(coord['y'][i])
                predictions['label'].append(label)
                predictions['confidence'].append(classification[0][label])
        confidence = confidence*percentage

    if(verbosity > 0):
        print "Number of found signs: ", len(predictions['x'])

    if(plot):
        rgb = [0, 0, 0]
        new_image = image[:, :, np.newaxis] + rgb
        # Create a legend at the bottom:
        classes_per_row = len(image[0])/30
        no_of_additional_rows = no_of_classes/classes_per_row + 1
        rows = np.zeros((no_of_additional_rows*30, len(image[0]), 3), dtype='f')
        rows.fill(255)
        new_image = np.concatenate((new_image, rows), 0)

        # for i in range(0, no_of_classes*30):
        #     row = []
        #     for j in range(0, len(image[0])):
        #         row.append(rgb)
        #     new_image = np.concatenate((new_image, row), 0)

        for i in range(0, len(predictions['x'])):
            create_rectangle(new_image, predictions['x'][i],
                    predictions['y'][i], predictions['label'][i],
                    image_width, image_width)
        create_legend(new_image, len(image), cuneiform, cuneiform_label,
                image_width, image_width, False)
        misc.imsave('labeled_cuneiform.jpg', new_image)


def simple_cunei():
    """Very basic test from
    https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners
    """
    if(orig):
        image_width = 128
    else:
        image_width = 28
    image_size = image_width*image_width
    cuneiform1, cuneiform_label1 = load_images()
    cuneiform, cuneiform_label = extend_set(cuneiform1,
            cuneiform_label1, extension, False)

    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, image_size])
    W = tf.Variable(tf.zeros([image_size, no_of_classes]))
    b = tf.Variable(tf.zeros([no_of_classes]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    ### Training
    y_ = tf.placeholder(tf.float32, [None, no_of_classes])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
            reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    ### Launching
    sess = tf.Session()
    sess.run(init)
    # for epoch_i in range(n_epochs):
    for batch_i in range(len(cuneiform) // batch_size):
        batch_xs = np.reshape(cuneiform[
                batch_i*batch_size:(batch_i+1)*batch_size],
                [batch_size, image_size])
        batch_ys = np.reshape(cuneiform_label[
                batch_i*batch_size:(batch_i+1)*batch_size],
                [batch_size, no_of_classes])
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    ### Evaluating
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if(verbosity > 0):
        print "\nAccuracy of our model:"
        n_examples = 20
        print(sess.run(accuracy, feed_dict={x: cuneiform[-n_examples:],
                y_: cuneiform_label[-n_examples:]}))
    classify('data/cuneiform_resized_full.png', sess, image_width, x, y,
            cuneiform, cuneiform_label)


def deep_cunei():
    """More advanced test from
    https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
    """
    ### Weight Initialization

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    ### Convolution and Pooling

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    # def conv2d_2(x, W, stride):
    #     return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
    #             padding='SAME')
    #
    # def max_pool_2x2_2(x, pool_size, stride):
    #     return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
    #                         strides=[1, stride, stride, 1], padding='SAME')

    if(orig):
        image_width = 128
    else:
        image_width = 28
    image_size = image_width*image_width
    cuneiform1, cuneiform_label1 = load_images()
    cuneiform, cuneiform_label = extend_set(cuneiform1,
                                            cuneiform_label1, extension, False)

    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, image_size])
    y_ = tf.placeholder(tf.float32, shape=[None, no_of_classes])
    W = tf.Variable(tf.zeros([image_size,no_of_classes]))
    b = tf.Variable(tf.zeros([no_of_classes]))
    sess.run(tf.initialize_all_variables())

    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                                  reduction_indices=[1]))
    ### 1st Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,image_width,image_width,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    ### 2nd Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    ### Densely Connected Layer
    # changed 32 to 7
    if(orig):
        W_fc1 = weight_variable([32 * 32 * 64, 1024])
    else:
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    if(orig):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
    else:
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    ### Readout Layer
    W_fc2 = weight_variable([1024, no_of_classes])
    b_fc2 = bias_variable([no_of_classes])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    ### Training and Evaluating
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                  reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(adam).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    # for epoch_i in range(n_epochs):
    for batch_i in range(len(cuneiform) // batch_size):
        batch_xs = np.reshape(cuneiform[
                batch_i*batch_size:(batch_i+1)*batch_size],
                [batch_size, image_size])
        batch_ys = np.reshape(cuneiform_label[
                batch_i*batch_size:(batch_i+1)*batch_size],
                [batch_size, no_of_classes])

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        if(verbosity > 0):
            if(batch_i%100 == 0):
                train_accuracy = accuracy.eval(feed_dict={
                        x:cuneiform[-batch_size:],
                        y_: cuneiform_label[-batch_size:], keep_prob: 0.5})
                print("step %d, training accuracy %g"%(batch_i, train_accuracy))
    classify('data/cuneiform_resized_minimal.png', sess, image_width, x, y,
            cuneiform, cuneiform_label)



def get_hog(block, block_width, n_bins=9, n_cells=4):
    """ Extract HOG-features of given window.
    1. Divide the image window into spatial regions (cells).
    2. For each cell compute the gradient histogram
    3. Combine the histograms.
    4. Normalize the histograms by using a histogram of the whole window
    (block)
    Normalization is done via L2-norm (with max 0.2)
    v/srqt(e*e+norm2(v)^2) where v is the array with all histograms.

    Parameters
    ----------
    block :
    block_width :
    n_bins :        Integer
                    Number of bins in a histogram for each cell. Only 9 are
                    supported at the moment
    n_cells :       Integer
                    Number of cells to divide the given block

    Returns
    -------
    histogram : List of floats

    """
    histograms = []
    stride = block_width/n_cells
    tmp_block = np.reshape(block, (block_width, block_width))
    # print np.shape(tmp_block)
    # print stride
    for(x, y, window) in sliding_window(tmp_block, stride, stride):
        # print np.shape(window)
        window = np.reshape(window, (stride*stride))
        histograms.append(get_histogram(window, stride, n_bins))
    norm2 = 0.0

    histograms = np.reshape(histograms, (len(histograms)*n_bins))
    for value in histograms:
        norm2 += value*value
    norm2 = np.absolute(norm2)
    # norm2 = np.sqrt(norm2)
    if(norm2 > 0.2):
        norm2 = 0.2
    factor =  1/(np.sqrt(norm2+np.exp(1)*np.exp(1)))
    histograms[:] = [x*factor for x in histograms]

    return histograms


def get_histogram(image, image_width, n_bins = 9):
    """Compute histogram to given data.
    http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
    Derivative Mask for x-axis:
            [-1 0 1]
    1/3  *  [-1 0 1]
            [-1 0 1]

    Derivative Mask for y-axis:
            [ 1  1  1]
    1/3  *  [ 0  0  0]
            [-1 -1 -1]
    Masks should reduce noise. Border pixels are set to 0.

    Parameters
    ----------
    image :         List
                    Flatted array of an image with values from 0 to 1.
    image_width :   Integer

    n_bins :        Integer
                    Number of bins for the histogram.
    """
    # http://codereview.stackexchange.com/questions/42763/histogram-of-oriented-gradients-hog-feature-detector-for-computer-vision-tr
    # Could be a version

    # Compute simple gradients for each pixel in each image in images.
    # gradients_in_image = []
    # angles = []
    # magnitudes = []
    # Use positive gradients only and store them in an histogram of 9 bins
    # 10, 30, 50, 70, 90, 110, 130, 150, 170
    histogram = np.zeros(n_bins)
    for i in range(0, len(image)):
        grad_x = 0
        grad_y = 0
        if(i%image_width != image_width-1 and i >= image_width and
                i<image_width*(image_width-1) and i%image_width != 0):
            grad_x = (image[i+1]+image[i-image_width-1]+image[i+image_width-1]
                      -image[i+1]-image[i-image_width+1]-image[i+image_width+1])

            grad_y = (image[i-image_width]+image[i-image_width-1]
                      +image[i-image_width+1]
                      -image[i+image_width]-image[i+image_width+1]
                      -image[i+image_width-1])
        # Compute the magnitude and angle of each gradient
        magnitude = np.sqrt(grad_x*grad_x+grad_y*grad_y)
        if(grad_x != 0 and grad_y != 0):
            angle = np.arctan(grad_y/grad_x)
        else:
            angle = 0
        # Convert from -pi/2,pi/2 to 0,pi
        if(angle<0):
            angle += np.pi
        # angles.append(angle)
        # magnitudes.append(magnitude)
        angle = angle*180/np.pi
        # If a gradient is between two bins, store part of its magnitude in both
        if(angle < 90):
            if(angle < 50):
                if(angle < 10):
                    # 0 < angle < 10
                    histogram[0] += magnitude
                elif(angle < 30):
                    # 30 < angle < 50
                    higher_part = (angle-30)/20
                    lower_part = 1-higher_part
                    histogram[1] += lower_part*magnitude
                    histogram[2] += higher_part*magnitude
                else:
                    # 10 < angle < 30
                    higher_part = (angle-10)/20
                    lower_part = 1-higher_part
                    histogram[0] += lower_part*magnitude
                    histogram[1] += higher_part*magnitude
            elif(angle < 70):
                # 50 < angle < 70
                higher_part = (angle-50)/20
                lower_part = 1-higher_part
                histogram[2] += lower_part*magnitude
                histogram[3] += higher_part*magnitude
            else:
                # 70 < angle < 90
                higher_part = (angle-70)/20
                lower_part = 1-higher_part
                histogram[3] += lower_part*magnitude
                histogram[4] += higher_part*magnitude
        elif(angle < 130):
            if(angle < 110):
                # 90 < angle < 110
                higher_part = (angle-90)/20
                lower_part = 1-higher_part
                histogram[4] += lower_part*magnitude
                histogram[5] += higher_part*magnitude
            else:
                # 110 < angle < 130
                higher_part = (angle-110)/20
                lower_part = 1-higher_part
                histogram[5] += lower_part*magnitude
                histogram[6] += higher_part*magnitude
        elif(angle < 150):
            # 130 < angle < 150
            higher_part = (angle-130)/20
            lower_part = 1-higher_part
            histogram[6] += lower_part*magnitude
            histogram[7] += higher_part*magnitude
        else:
            # 150 < angle <= 170
            higher_part = (angle-150)/20
            lower_part = 1-higher_part
            histogram[7] += lower_part*magnitude
            histogram[8] += higher_part*magnitude

    return histogram




    # image_size = image_width*image_width
    # histogram = np.zeros((n_cells*n_cells*n_bins)) # TODO: Is this right?
    # bin_range = (2*np.pi)/n_bins
    # cellx = image_width / n_cells  # width of each cell(division)
    # celly = image_width / n_cells  # height of each cell(division)

    # TODO: Check if this is correct
    # it = product(xrange(n_cells), xrange(n_cells), xrange(cellx), xrange(celly))
    # for m, n, i, j in it:
    #     # grad value
    #     grad = gradients_in_image[m * cellx + i, n * celly + j][0]
    #     # normalized grad value
    #     norm_grad = grad / image_size
    #     # Orientation Angle
    #     angle = gradients_in_image[m*cellx + i, n*celly+j][1]
    #     # (-pi,pi) to (0, 2*pi)
    #     if angle < 0:
    #         angle += 2 * pi
    #     nth_bin = floor(float(angle/bin_range))
    #     histogram[((m * n_divs + n) * n_bins + int(nth_bin))] += norm_grad
    # return histogram

    # bins = (angles[:] % (2 * np.pi) / bin_range).astype(int)
    # x, y = np.mgrid[:image_width, :image_width]
    # x = x * n_cells // image_width
    # y = y * n_cells // image_width
    # labels = (x * n_cells + y) * n_bins + bins
    # index = np.arange(n_cells*n_cells*n_bins)
    # histogram = scipy.ndimage.measurements.sum(magnitudes[0:], labels, index)
    # return histogram / image_size


def non_max_suppression(image, coord, box_width, threshold, labels):
    """identify overlapping boxes and pick one of them.
    Inspired by
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    This method checks each label independently.

    Parameters
    ----------
    image :     List
                Image with height, length and RGB-channel
    coord :     Dictionary with 'x' and 'y'
                The upper left corner of the bounding box.
    box_width : Int
                Width and height of the box.
    threshold : Float
                Threshold <= 1.
    labels :     List
                List of labels for each box. Overlapping boxes from different
                labels should not matter.

    Returns
    -------
    picked_boxes :    Dictionary with 'x' and 'y'
                      Dictionary of chosen boxes
    picked_labels :   List
                      Corresponding list of labels

    """
    if(len(labels) == 0):
        return []
    x_coord = np.asarray(coord['x'], dtype=np.float32)
    y_coord = np.asarray(coord['y'], dtype=np.float32)
    # if(coord['x'].dtype.kind == 'i'):
    #     coord['x'] = coord['x'].astype("float")
    # if(coord['y'].dtype.kind == 'i'):
    #     coord['y'] = coord['y'].astype("float")
    picked_boxes = {'x': [], 'y': []}

    picked_labels = []
    # Pick boxes for each label
    for i in range(0, no_of_classes):
        idx_picked_boxes = []
        # Need the coordinates of the bounding boxes
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for j in range(0,len(labels)):
            if(labels[j] == i):
                x1.append(x_coord[j])
                y1.append(y_coord[j])
                x2.append(x_coord[j]+28)
                y2.append(y_coord[j]+28)
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        area = (x2-x1+1) * (y2-y1+1)
        idxs = np.argsort(y2)

        while(len(idxs) > 0):
            last = len(idxs)-1
            j = idxs[last]
            idx_picked_boxes.append(j)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[j], x1[idxs[:last]])
            yy1 = np.maximum(y1[j], y1[idxs[:last]])
            xx2 = np.minimum(x2[j], x2[idxs[:last]])
            yy2 = np.minimum(y2[j], y2[idxs[:last]])

            # Compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have bigger overlapp
            idxs = np.delete(idxs, np.concatenate(([last],
                    np.where(overlap > threshold)[0])))

        for j in range(0, len(idx_picked_boxes)):
            idx = idx_picked_boxes[j]
            picked_boxes['x'].append(x1[idx])
            picked_boxes['y'].append(y1[idx])
            picked_labels.append(i)

    return picked_boxes, picked_labels


def non_max_suppression2(image, coord, box_width, threshold, labels):
    """identify overlapping boxes and pick one of them.
    Inspired by
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    The label of each box is ignored while choosing the box.

    Parameters
    ----------
    image :     List
                Image with height, length and RGB-channel
    coord :     Dictionary with 'x' and 'y'
                The upper left corner of the bounding box.
    box_width : Int
                Width and height of the box.
    threshold : Float
                Threshold <= 1.
    labels :     List
                List of labels for each box. Overlapping boxes from different
                labels should not matter.

    Returns
    -------
    picked_boxes :    Dictionary with 'x' and 'y'
                      Dictionary of chosen boxes
    picked_labels :   List
                      Corresponding list of labels

    """
    if(len(labels) == 0):
        return []
    x_coord = np.asarray(coord['x'], dtype=np.float32)
    y_coord = np.asarray(coord['y'], dtype=np.float32)
    # if(coord['x'].dtype.kind == 'i'):
    #     coord['x'] = coord['x'].astype("float")
    # if(coord['y'].dtype.kind == 'i'):
    #     coord['y'] = coord['y'].astype("float")
    picked_boxes = {'x': [], 'y': []}

    picked_labels = []
    # Pick boxes for each label

    idx_picked_boxes = []
    # Need the coordinates of the bounding boxes
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for j in range(0,len(labels)):
        x1.append(x_coord[j])
        y1.append(y_coord[j])
        x2.append(x_coord[j]+28)
        y2.append(y_coord[j]+28)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    area = (x2-x1+1) * (y2-y1+1)
    idxs = np.argsort(y2)

    while(len(idxs) > 0):
        last = len(idxs)-1
        j = idxs[last]
        idx_picked_boxes.append(j)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[j], x1[idxs[:last]])
        yy1 = np.maximum(y1[j], y1[idxs[:last]])
        xx2 = np.minimum(x2[j], x2[idxs[:last]])
        yy2 = np.minimum(y2[j], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have bigger overlapp
        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > threshold)[0])))

    for j in range(0, len(idx_picked_boxes)):
        idx = idx_picked_boxes[j]
        picked_boxes['x'].append(x1[idx])
        picked_boxes['y'].append(y1[idx])
        picked_labels.append(labels[idx])

    return picked_boxes, picked_labels


def hog_cunei(fit):
    """Use histogram of oriented gradients for training. See
    http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/
    1. We extract the HOG features for each label.
    2. Extract HOG features of many negative examples (images without any
    letter which has been labeled.)
    3. Train a SVM with negative and positive samples.
    4. Use sliding window on a small image to detect the letters. Each window
    needs its HOG-features. Record false-
    positives and their probability.
    5. Use false-positives as new samples (ordered by the confidence).
    Iterate 4 and 5 although it might not help much.
    6. Use the trained classifier on the full dataset like in step 4. Take
    the best results.
    7. Use non-maximum suppression to remove redundant and overlapping
    bounding boxes.
    http://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
    """
    image_width = 28
    # We use high def here
    orig = False
    # image_size is the number of features for the svm
    image_size = image_width*image_width

    cuneiform1, cuneiform_label1 = load_images()
    cuneiform, cuneiform_label = extend_set(cuneiform1,
            cuneiform_label1, extension, True)
    cuneiform_hog = []

    # 1. We extract the HOG features for each label.
    # 2. Extract HOG features of many negative examples
    for image in cuneiform:
        cuneiform_hog.append(get_hog(image, image_width))


    # 3. Train svm with sklearn. Use as many svms as labels.
    # svm uses one-vs-one multiclass-svm
    model = svm.SVC(kernel=fit, probability=True)
    model.fit(cuneiform_hog, cuneiform_label)
    print "The score: ", model.score(cuneiform_hog, cuneiform_label)

    # Winner-takes-all: Classifier with highest output assigns the class.
    # models = []
    # for i in range(0,no_of_classes):
    #     # Add support vector classifier. Might try rbf, poly or sigmoid
    #     models.append(svm.svc(kernel='linear', c=1, gamma=1, probability=True))
    # 4. Detect letters
    image = misc.imread('data/cuneiform_resized_full.png')
    data = []
    coord = {'x': [], 'y': []}
    for(x_coord, y_coord, window) in sliding_window(image, 4, image_width):
        # Ignore windows with different sizes at en edge.
        if window.shape[0] != image_width or window.shape[1] != image_width:
            continue

        # Add the window to our dataset
        # window = window[:,:,0]
        window = np.reshape(window, image_width*image_width)
        window = window.astype('f')
        window = window - 255
        window = np.absolute(window)
        window = window/255
        data.append(get_hog(window, image_width))
        coord['x'].append(x_coord)
        coord['y'].append(y_coord)

    # labels = model.predict(data)
    # TODO: Delete labels with bad probability.
    lables_tmp = model.predict_proba(data)

    labels = []
    found_coord = {'x': [], 'y': []}
    global confidence
    max_confidence = confidence
    for i in range(0,len(lables_tmp)):
        idx = np.argmax(lables_tmp[i])
        if(lables_tmp[i][idx] > confidence):
            labels.append(idx)
            found_coord['x'].append(coord['x'][i])
            found_coord['y'].append(coord['y'][i])
            if(max_confidence < lables_tmp[i][idx]):
                max_confidence = lables_tmp[i][idx]

    if(verbosity > 0):
        print "Found labels: ", len(labels)

    percentage = confidence
    if(max_confidence > confidence):
        max_confidence = confidence
    confidence = max_confidence
    while ( len(labels) < 30 and percentage > 0):
        labels = []
        found_coord = {'x': [], 'y': []}
        print "Trying another confidence with ", confidence
        for i in range(0,len(lables_tmp)):
            idx = np.argmax(lables_tmp[i])
            if(lables_tmp[i][idx] > confidence):
                labels.append(idx)
                found_coord['x'].append(coord['x'][i])
                found_coord['y'].append(coord['y'][i])
        confidence = confidence*percentage

    if(verbosity > 0):
        print "Found labels: ", len(labels)

    # 5. Is done by hand

    if(plot):
        rgb = [0, 0, 0]
        new_image = image[:, :, np.newaxis] + rgb
        # Create a legend at the bottom:
        classes_per_row = len(image[0])/30
        if(verbosity > 1):
            print "classes_per_row ", classes_per_row
            print "no_of_classes ", no_of_classes
            print "shape of image: ", np.shape(image)
            print "div ", len(image[0])/30
            print "len ", len(image[0])
        no_of_additional_rows = no_of_classes/classes_per_row + 1
        rows = np.zeros((no_of_additional_rows*30, len(image[0]), 3), dtype='f')
        rows.fill(255)
        new_image = np.concatenate((new_image, rows), 0)
        if(verbosity > 1):
            verbo_image = new_image.copy()
        # 7. Use non-maximum suppression (or Canny-algorithm)
        # to delete overlapping boxes.
        picked_boxes, picked_labels = non_max_suppression2(new_image,
                found_coord, image_width, 0.1, labels)
        if(verbosity > 0):
            print "Boxes before: ", len(labels), " and after: ", len(picked_labels)
        # Draw the boxes
        for i in range(0, len(picked_labels)):
            create_rectangle(new_image, picked_boxes['x'][i],
                    picked_boxes['y'][i], picked_labels[i],
                    image_width, image_width)

        create_legend(new_image, len(image), cuneiform, cuneiform_label,
                image_width, image_width, True)
        misc.imsave('labeled_cuneiform_HOG_SVM.jpg', new_image)

        ##############save all boxes as image########################################
        # for i in range(0,len(picked_labels)):
        #     imName = 'windows/' + str(picked_labels[i]+1) + '_' + str(i) + '.png'
        #     img = np.zeros((28,28), dtype = int)
        #
        #     # tmp = np.absolute(image[int(picked_boxes['y'][i])][int(picked_boxes['x'][i])]-1)
        #     # tmp = tmp*255
        #
        #     starty = int(picked_boxes['y'][i])
        #     startx = int(picked_boxes['x'][i])
        #     for y_c in range(starty, starty+28):
        #         for x_c in range(startx, startx+28):
        #
        #             img[y_c-starty][x_c-startx] = image[y_c][x_c]
        #     misc.imsave(imName, img)
        ################################################################################

        if(verbosity > 1):
            for i in range(0, len(labels)):
                create_rectangle(verbo_image, found_coord['x'][i],
                        found_coord['y'][i], labels[i],
                        image_width, image_width)

            create_legend(verbo_image, len(image), cuneiform, cuneiform_label,
                    image_width, image_width, True)
            misc.imsave('labeled_cuneiform_HOG_SVM_debug.jpg', verbo_image)


if __name__ == '__main__':
    parser = ArgumentParser(
    description=
    '''Use autoencoder to learn the classes of MNIST. Also simple neural
nets for classifying cuneiform scripts.''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--method', type=int, required=True, default=1,
                        help=
'''Choose the method to use by giving an integer.
Method 1: Basic test with AdamOptimizer and MNIST.
Method 2: Not yet implemented.
Method 3: Simple test with softmax regression and cuneiform script.
Method 4: More advanced than method 3 with multiple layers.
Method 5: Extract HOG-features from a test set and use a multi-class
support vector machine to classify a cuneiform script.
''')
    parser.add_argument('-l', '--learning_rate', type=float, required=False,
                        default=0.001,
                        help='''Set the learning rate (bigger than 0).
Default is 0.001.''')
    parser.add_argument('-b', '--batch_size', type=int, required=False,
                        default=50,
                        help=
                        '''Set the batch_size (bigger than 0).
Default is 50.''')
    parser.add_argument('-n', '--n_epochs', type=int, required=False,
                        default=10,
                        help=
                        '''Set the number of epochs (bigger than 0) to fit all
training data. Default is 10.''')
    parser.add_argument('-p', '--plot', dest='plot', required=False,
                        action='store_true',
                        help=
                        '''Plot an example reconstruction. Use -f or
--filename to specify an output name.''')
    parser.add_argument('-f', '--filename', type=str, required=False,
                        default='Autoencoder',
                        help=
                        '''Specify a filename to save a plot. Requires -p
or --plot to be True.''')
    parser.add_argument('-c', '--confidence', type=float, required=False,
                        default = 0.9999,
                        help= '''Set the confidence for found signs. Default is
0.9999. Lower confidence leads to more false
positives. Only applies if -p or --plot is used.''')
    parser.add_argument('-v', '--verbose', type=int, required=False, default=1,
                        help=
                        '''Set verbosity.
0 = Minimal output.
1 = Some output like current error and epoch. (Default)
2 = More output for debugging like start of the test.''')
    parser.add_argument('-e', '--extension', type=int, required=False,
                        default=1000,
                        help=
                        '''Extend the dataset by the given number.
The data will be copied randomly with some rotation.
                        ''')
    parser.add_argument('-g', '--gaussian', dest='gaussian', required=False,
                        action='store_true',
                        help=
                        '''Add some gaussian noise (mean=0, sd=0.05) to your
example data.''')
    parser.add_argument('-a', '--adam', type=float, required=False,
                        default=1e-4,
                        help='''Factor for AdamOptimizer in method 4. Default
is 1e-4.''')
    parser.add_argument('--fit', type=str, required=False,
                        default='poly',
                        help=
                        '''Define a fit for svm. Possible fits are:
linear, rbf, poly (default), sigmoid''')
    parser.set_defaults(plot=False, gaussian = False)
    args = parser.parse_args()
    method = args.method
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    verbosity = args.verbose
    plot = args.plot
    filename = args.filename
    extension = args.extension
    confidence = args.confidence
    gaussian = args.gaussian
    adam = args.adam
    fit = args.fit

    if(learning_rate < 0):
        print "Your learning rate is smaller than 0. It is set to 0.001."
        learning_rate = 0.001
    if(batch_size < 0):
        print "Your batch size is smaller than 0. It is set to 50."
        learning_rate = 50
    if(n_epochs < 0):
        print "Your number of epochs is smaller than 0. It is set to 1."
        n_epochs = 1
    if(method == 1):
        if(verbosity > 0):
            print "Starting method 1: Basic test with AdamOptimizer."
            print "Learning rate: ", learning_rate
            print "Batch size: ", batch_size
            print "Number of epochs: ", n_epochs
            print "Plot samples?: ", plot
            print "--------------------------------"
        test_mnist()
    elif(method == 2):
        if(verbosity > 0):
            print "Starting method 2: "
            print "Number of epochs: ", n_epochs
            print "--------------------------------"
        test_mnist2()
    elif(method == 3):
        if(verbosity > 0):
            print "Starting method 3: Basic test with cuneiform scripts."
            print "Learning rate: ", learning_rate
            print "Batch size: ", batch_size
            print "Plot samples?: ", plot
            print "verbosity: ", verbosity
            print "--------------------------------"
        simple_cunei()
    elif(method == 4):
        if(verbosity > 0):
            print "Starting method 4: ."
            print "Learning rate: ", learning_rate
            print "Batch size: ", batch_size
            print "Plot samples?: ", plot
            print "verbosity: ", verbosity
            print "--------------------------------"
        deep_cunei()
    elif(method == 5):
        if(verbosity > 0):
            print "Starting method 5: HOG."
            print "Plot samples?: ", plot
            print "verbosity: ", verbosity
            print "--------------------------------"
        hog_cunei(fit)
    else:
        print "Error: No such method: ", method
        print "Use -h for help"
        exit()
