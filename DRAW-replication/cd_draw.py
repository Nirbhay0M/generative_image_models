#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
	python draw.py --data_dir=/tmp/draw --read_attn=True --write_attn=True

Author: Eric Jang
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os

tf.flags.DEFINE_string("data_dir","./data/", "")
tf.flags.DEFINE_string("save_suffix","cd_double_conv", "")
tf.flags.DEFINE_boolean("read_attn", False, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",False, "enable attention for writer")
tf.flags.DEFINE_boolean("restore",False, "restore model")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ## 

A,B = 28,28 # image width,height
img_size = B*A # the canvas size
n_channels = 1 # Color channels
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
kernel_height = 5
kernel_width = 5
# n_filters = 32
read_n = 5 # read glimpse grid width/height
write_n = 5 # write glimpse grid width/height
read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n if FLAGS.write_attn else img_size
z_size=10 # QSampler output size
T=10 # MNIST generation sequence length
batch_size=100 # training minibatch size
train_iters=10000
learning_rate=1e-3 # learning rate for optimizer
eps=1e-8 # epsilon for numerical stability
display_step = 1

## BUILD MODEL ## 

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size)) # input (batch_size * img_size)
e=tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
lstm_dec = tf.nn.rnn_cell.LSTMCell(dec_size, state_is_tuple=True) # decoder Op

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy

def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

## READ ## 
def read_no_attn(x,x_hat,h_dec_prev=None):
    stride = 2
    shape = (batch_size,B,A,1)
    # print "Shape:",shape
    x_1,shape = conv2d("conv2d_x_1",x,stride,32,shape2D=shape)
    # print "Shape:",shape
    x_2,shape = conv2d("conv2d_x_2",x_1,stride,32,shape2D=shape)
    # print "Shape:",shape
    x_new = tf.concat(1,[x,x_1,x_2])

    # assert False
    shape = (batch_size,B,A,1)
    x_hat_1,shape = conv2d("conv2d_x_hat_1",x_hat,stride,64,shape2D=shape)
    x_hat_2,shape = conv2d("conv2d_x_hat_2",x_hat_1,stride,64,shape2D=shape)
    x_hat_new = tf.concat(1,[x_hat,x_hat_1,x_hat_2])

    return tf.concat(1,[x_new,x_hat_new])

def read_attn(x,x_hat,h_dec_prev):
    Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n)
    return tf.concat(1,[x,x_hat]) # concat along feature axis

read = read_attn if FLAGS.read_attn else read_no_attn

## ENCODE ## 
def encode(state,input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder",reuse=DO_SHARE):
        return lstm_enc(input,state)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu",reuse=DO_SHARE):
        mu=linear(h_enc,z_size)
    with tf.variable_scope("sigma",reuse=DO_SHARE):
        logsigma=linear(h_enc,z_size)
        sigma=tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)

## DECODER ## 
def decode(state,input):
    with tf.variable_scope("decoder",reuse=DO_SHARE):
        return lstm_dec(input, state)

## WRITER ## 
def write_no_attn(h_dec,conv_shape=None):
    with tf.variable_scope("write",reuse=DO_SHARE):
        return linear(h_dec,img_size)

    # # print "write got conv_shape:",conv_shape
    # # conv_shape is (batch_size, B', A', n_filters)
    # with tf.variable_scope("write",reuse=DO_SHARE):
    #     decoded_img_size = conv_shape[1]*conv_shape[2]*conv_shape[3]
    #     # return linear(h_dec_deconv,img_size)
    #     img_decoded = linear(h_dec,decoded_img_size)
    #     img_deconv = conv2d_transpose(img_decoded,2,conv_shape,
    #                         output_shape=[batch_size,B,A,1])
    #     return tf.reshape(img_deconv,[batch_size,-1])

def write_attn(h_dec):
    with tf.variable_scope("writeW",reuse=DO_SHARE):
        w=linear(h_dec,write_size) # batch x (write_n*write_n)
    N=write_n
    w=tf.reshape(w,[batch_size,N,N])
    Fx,Fy,gamma=attn_window("write",h_dec,write_n)
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.batch_matmul(Fyt,tf.batch_matmul(w,Fx))
    wr=tf.reshape(wr,[batch_size,B*A])
    #gamma=tf.tile(gamma,[1,B*A])
    return wr*tf.reshape(1.0/gamma,[-1,1])

write=write_attn if FLAGS.write_attn else write_no_attn

def conv2d(scope,x_input, strides, n_filters,shape2D):
    _,_height,_width,_channels = shape2D
    # original_shape = tf.shape(x_input)
    # print "conv2d got input: ",x_input
    x = tf.reshape(x_input,[batch_size,_height,_width,_channels])

    with tf.variable_scope(scope,reuse=DO_SHARE):
        W = tf.get_variable("W_kernel",[kernel_height,kernel_width,_channels,n_filters],
                                initializer=tf.random_normal_initializer())
        b = tf.get_variable("b_kernel",[n_filters],
                                initializer=tf.random_normal_initializer())

        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        # x has shape: (batch_size, B', A', n_filters)
        output_shape = [ int(s) for s in x.get_shape() ]
        # print "output_shape:",output_shape

        x_out = tf.reshape(x,[batch_size,-1])

        return tf.nn.relu(x_out),output_shape

def conv2d_transpose(x_input, strides, conv_shape, output_shape):
    x = tf.reshape(x_input,conv_shape)

    # with tf.variable_scope(scope,reuse=DO_SHARE):
    W = tf.get_variable("W_kernel_deconv",[kernel_height,kernel_width,1,n_filters],
                        initializer=tf.random_normal_initializer())
    
    x_deconv = tf.nn.conv2d_transpose(x, W, strides=[1, strides, strides, 1],
                            output_shape=output_shape, padding='SAME')
    x_output = tf.reshape(x_deconv,[batch_size,-1])

    b = tf.get_variable("b_kernel_deconv",[B*A],
                        initializer=tf.random_normal_initializer())

    x_output = tf.nn.bias_add(x_output, b)

    # return tf.nn.sigmoid(x_output)
    return tf.nn.relu(x_output)
    # return x_output

## STATE VARIABLES ## 

cs=[0]*T # sequence of canvases
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ## 

# construct the unrolled computational graph
for t in range(T):
    c_prev = tf.zeros((batch_size,img_size)) if t==0 else cs[t-1]
    x_hat=x-tf.sigmoid(c_prev) # error image
    # r=read(x,x_hat,h_dec_prev)
    r =read_no_attn(x,x_hat)
    h_enc,enc_state=encode(enc_state,tf.concat(1,[r,h_dec_prev]))
    z,mus[t],logsigmas[t],sigmas[t]=sampleQ(h_enc)
    h_dec,dec_state=decode(dec_state,z)
    cs[t]=c_prev+write(h_dec) # store results
    h_dec_prev=h_dec
    DO_SHARE=True # from now on, share variables

## LOSS FUNCTION ## 

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons=tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-T*.5 # each kl term is (1xminibatch)
KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz=tf.reduce_mean(KL) # average over minibatches

cost=Lx+Lz

# for v in tf.all_variables():
#     print("%s : %s" % (v.name,v.get_shape()))
# assert False

## OPTIMIZER ## 

optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(cost)
mean_grads = []
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
        mean_grads.append(tf.reduce_mean(grads[i]))
train_op=optimizer.apply_gradients(grads)

## RUN TRAINING ## 

data_directory = os.path.join(FLAGS.data_dir, "mnist")
if not os.path.exists(data_directory):
	os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data
ckpt_file=os.path.join(FLAGS.data_dir,"save_"+str(FLAGS.save_suffix)+".ckpt")

fetches=[]
fetches.extend([Lx,Lz,train_op])
fetches.extend([mean_grads])
Lxs=[0]*train_iters
Lzs=[0]*train_iters

sess=tf.InteractiveSession()

saver = tf.train.Saver() # saves variables learned during training
tf.initialize_all_variables().run()

if FLAGS.restore:
    saver.restore(sess, ckpt_file) # to restore from model, uncomment this line
else:
    print "Beginning training!"
    grad_log = open("./tmp/GradLog.txt","w")

    for i in range(train_iters):
    	xtrain,_=train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
    	feed_dict={x:xtrain}
    	results=sess.run(fetches,feed_dict)
    	Lxs[i],Lzs[i],_,m_gradients=results
    	if i%display_step==0:
            print("iter=%d : Lx: %f Lz: %f" % (i,Lxs[i],Lzs[i]))
            grad_log.write(str(m_gradients)+"\n")
    grad_log.close()
## TRAINING FINISHED ## 

canvases=sess.run(cs,feed_dict) # generate some examples
canvases=np.array(canvases) # T x batch x img_size

out_file=os.path.join(FLAGS.data_dir,"cd_draw_data"+str(FLAGS.save_suffix)+".npy")
np.save(out_file,[canvases,Lxs,Lzs])
print("Outputs saved in file: %s" % out_file)

save_path = saver.save(sess,ckpt_file)
print("Model saved in file: %s" % save_path)

sess.close()

print('Done drawing! Have a nice day! :)')
