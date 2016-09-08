#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import argparse,argcomplete

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Logging
import logging
logging.basicConfig(filename='logfile.log',level=logging.DEBUG,
    format='%(asctime)s :: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
restore_vars = False

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Op to save and restore variables
saver = tf.train.Saver()

def assignArgs(args):
    global learning_rate
    global training_iters
    global batch_size
    global display_step
    global restore_vars

    # learning_rate   = args.learning_rate 
    # training_iters  = args.training_iters
    # batch_size      = args.batch_size
    # display_step    = args.display_step

    restore_vars    = bool(args.restore_vars)

    return

def parseArgs():
    parser = argparse.ArgumentParser(description="Architecture Parameters")

    #--- Restoring saved variables
    parser.add_argument('-res','--restore-vars',default=0,type=int)
    
    argcomplete.autocomplete(parser)
    logging.info("Parsing arguments.")
    args = parser.parse_args()

    logging.debug("Args:"+str(args))
    assignArgs(args)

    return args

if __name__=='__main__':
    logging.info("Main Begins!")
    parseArgs()

    save_model_name = "model_autoenc.ckpt"
    
    # Launch the graph
    with tf.Session() as sess:
        if restore_vars:
            saver.restore(sess,"./tmp/"+save_model_name)
            logging.info("Model Restored!")

        else:
            sess.run(init)
            total_batch = int(mnist.train.num_examples/batch_size)
            # Training cycle
            for epoch in range(training_epochs):
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1),
                          "cost=", "{:.9f}".format(c))

            print("Optimization Finished!")

            # Save variables to a file
            save_path = saver.save(sess,"./tmp/"+save_model_name)
            print("Model saved in file: %s" % save_path)

        # Applying encode and decode over test set
        encode_decode = sess.run(
            y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()