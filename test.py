import tensorflow as tf
from CRFasRNNcell import CRFasRNNcell
import numpy as np
import math
import cv2
from getPascalVOC2012 import genVOC2012filelist,load_Y,load_X

tf.app.flags.DEFINE_string("data_dir","img/","Image directory.")
tf.app.flags.DEFINE_string("out_dir", "result/", "Output directory.")
tf.app.flags.DEFINE_integer("conv_channel", 32, "Size to convolution.")
tf.app.flags.DEFINE_boolean("use_peepholes", True, "Using peepholes or not.")
tf.app.flags.DEFINE_float("cell_clip", None, "Value of cell clipping.")
tf.app.flags.DEFINE_float("forget_bias", 1.0, "Value of forget bias.")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_integer("train_step", 10000, "Num to train.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Num of batch size.")

FLAGS = tf.app.flags.FLAGS

IMG_SIZE = [224,224]
KERNEL_SIZE = [10, 10]
STRIDE = [1,1,1,1]

def inference(images,isdynamic=True):
    cell=CRFasRNNcell(IMG_SIZE,KERNEL_SIZE,)
    if(isdynamic):
        outputs, state = tf.nn.dynamic_rnn(cell,inputs=images, dtype=tf.float32)
    else:
        outputs, state = tf.nn.rnn(cell=cell, inputs=images, dtype=tf.float32)
    last_output=outputs[-1]

    return result

def loss(result, correct_image):
    return tf.reduce_mean(tf.abs(result-correct_image))

def train(images):
    return tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(error)

def predict(images):
    pass

def main(ddir="VOC"):
    if(ddir=="VOC"):
        IMG_SIZE = [224,224]
        xfiles,yfiles=genVOC2012filelist(ddir,True,rate=0.9)

    input_ph = tf.placeholder(tf.float32,[None, IMG_SIZE[0], IMG_SIZE[1], 3])
    y = tf.placeholder(tf.float32,[None, IMG_SIZE[0], IMG_SIZE[1], 3])

    result = inference(input_ph)
    error = loss(result, y)
    train_step = train(error)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        test_feed = {}

        test_feed[input_ph]= np.asarray([cv2.imread(xfiles[i])/255. for i in range(batch_size)])
        test_feed[y] =       np.asarray([cv2.imread(yfiles[i])/255. for i in range(batch_size)])

        sess.run(init_op)
        for step in range(FLAGS.train_step):
            feed_dict = {}
            target = [random.randint(0,100) for i in range(FLAGS.batch_size)]
            feed_dict[input_ph]=np.asarray([cv2.imread(xfiles[batch_size+i])/255. for i in target])
            feed_dict[y]       =np.asarray([cv2.imread(yfiles[batch_size+i])/255. for i in target])

            print("step%d training"%step)
            sess.run(train_step, feed_dict=feed_dict)

            if (step+1) % 10 == 0:
                created, error_val = sess.run([result, error], feed_dict=test_feed)
                print("step%d loss: %f" % (step, error_val))
                cv2.imwrite(FLAGS.out_dir+"step"+str(step)+".png", created[0] * 255.)



if __name__ == "__main__":
    main()
