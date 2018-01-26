import numpy as np
import os
import math
#os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64:/usr/lib/nvidia-387/"
import tensorflow as tf

lst = os.listdir('~/Desktop/images/')
#print(len(lst)) # 40 if we have saved 40 images
sess = tf.Session()
data_list_train = []
images=[]
for i in range(len(lst)):
    abspath = os.path.join('~/Desktop/images/',lst[i])
    #print (abspath)
    # refer https://www.tensorflow.org/versions/r1.5/api_docs/python/tf/image/decode_jpeg
    image = tf.image.decode_jpeg(tf.read_file(abspath),channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 200,300) # all images are not of same size
    image = (sess.run(image)) # 40*200*300*3
    #print (image.shape) # 200,300,3
    r = image[:,:,0].flatten()
    g = image[:,:,1].flatten()
    b = image[:,:,2].flatten()
    arr = np.array(list(r)+list(g)+list(b), dtype=np.uint8)
    images.append(arr)
    #print (np.array(images).shape) ?,180000
data = np.row_stack(images) 
print (data.shape) # 40, 180000(=200*300*3)
#print (type(images))   
#data_list_train.append(images)
#x = np.concatenate(np.array(images), axis=0)
#print (x.shape) # 7200000 = 180000*40
########################################################
# set random seed
tf.set_random_seed(123)
np.random.seed(123)

output_dir = "~/Desktop/out/out1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# construct the model directory template name
model_dir = os.path.join("~/Desktop/out/model_pixel_cnn" + '%s')
# iterate until we find an index that hasn't been taken yet.
i = 0
while os.path.exists(model_dir % i):
    i += 1
model_dir = model_dir % i
# create the folder
os.makedirs(model_dir)   
########################################################
# calculate input, output and conv2d_out_logits
########################################################
input_shape = [None, 180000, 1] # height=200*300*3
inputs = tf.placeholder(tf.float32, input_shape)

# Refer https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
h1_units=7 # hidden1
h2_units=5 # hidden2
out_units=2 # softmax
with tf.name_scope('h1'):
    w = tf.Variable(tf.truncated_normal([180000, h1_units], stddev=1.0/math.sqrt(float(180000))), name='weights')
    b = tf.Variable(tf.zeros([h1_units]), name='biases')
with tf.name_scope('h2'):
    w = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=1.0/math.sqrt(float(h1_units))), name='weights')
    b = tf.Variable(tf.zeros([h2_units]), name='biases')
with tf.name_scope('softmax_linear'):
    w = tf.Variable(tf.truncated_normal([h2_units, out_units], stddev=1.0/math.sqrt(float(h2_units))), name='weights')
    b = tf.Variable(tf.zeros([out_units]), name='biases')
#######################################################
# TBD how to calculate logits?
# Refer https://github.com/philkuz/PixelRNN/blob/master/pixelrnn.ipynb
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=inputs, name='loss'))

optimizer = tf.train.RMSPropOptimizer(0.1) # learning rate 1e-3
# Function compute_gradients(loss, <list of variables>) 
# Computes the gradients for a list of variables. 
#returns  a list of tuples (gradient, variable).
grads_and_vars = optimizer.compute_gradients(loss)
grad_clip=1.
#The optimizer yields None when gradients are null instead of zeroes.
new_grads_and_vars = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -grad_clip, grad_clip), gv[1]], grads_and_vars)
#new_grads_and_vars = [(tf.clip_by_value(gv[0], -grad_clip, grad_clip), gv[1]) for gv in grads_and_vars]

optim = optimizer.apply_gradients(new_grads_and_vars)
#######################################################
init = tf.global_variables_initializer()
sess.run(init)
print("Start training")
total_train_costs = []
for idx in range(1,40,1): # each batch is of size 1
  _, cost = sess.run([optim, loss], feed_dict={ inputs : data[idx] })
  total_train_costs.append(cost)
avg_train_cost = np.mean(total_train_costs)
print("Average Training Cost is" +str(avg_train_cost))
########################################################
sess.close()
