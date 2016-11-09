#!/usr/bin/env python

import argparse,argcomplete

import tensorflow as tf

# MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Logging
import logging
logging.basicConfig(level=logging.DEBUG, 
    format='%(asctime)s :: %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p')#,
    #filename='logfile.log')

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10
restore_vars = False

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]),name='wc1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]),name='wc2'),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 784]),name='wd1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]),name='w_out')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]),name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]),name='bc2'),
    'bd1': tf.Variable(tf.random_normal([784]),name='bd1'),
    'out': tf.Variable(tf.random_normal([n_classes]),name='b_out')
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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

    learning_rate   = args.learning_rate 
    training_iters  = args.training_iters
    batch_size      = args.batch_size
    display_step    = args.display_step

    restore_vars    = bool(args.restore_vars)

    return

def parseArgs():
    parser = argparse.ArgumentParser(description="Architecture Parameters")

    #--- Restoring saved variables
    parser.add_argument('-res','--restore-vars',default=0,type=int)

    #--- Training Params
    parser.add_argument('-lr','--learning-rate',default=0.001,type=float)
    parser.add_argument('-bs','--batch-size',default=128,type=int)
    parser.add_argument('-it','--training-iters',default=10000,type=int)
    parser.add_argument('-disp','--display-step',default=10,type=int)

    #--- Model Params
    parser.add_argument('-ipd','--ip-dropout',default=0.75,type=float)

    argcomplete.autocomplete(parser)
    logging.info("Parsing arguments.")
    args = parser.parse_args()

    logging.debug("Args:"+str(args))
    assignArgs(args)

    return args

if __name__=='__main__':
    logging.info("Main Begins!")

    parseArgs()

    save_model_name = "model_cnn.ckpt"

    # Launch the graph
    with tf.Session() as sess:

        if restore_vars:    
            saver.restore(sess,"./tmp/"+save_model_name)
            logging.info("Model Restored!")

        else:
            sess.run(init)

            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                               keep_prob: dropout})
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                      y: batch_y,
                                                                      keep_prob: 1.})
                    print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc)
                step += 1
            print "Optimization Finished!"

            # Save variables to a file
            save_path = saver.save(sess,"./tmp/"+save_model_name)
            print("Model saved in file: %s" % save_path)

        # Calculate accuracy for 256 mnist test images
        print "Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                          y: mnist.test.labels[:256],
                                          keep_prob: 1.})