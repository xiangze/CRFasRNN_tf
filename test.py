import tensorflow as tf
from CRFasRNNcell import CRFasRNNcell
import numpy as np
import math
import cv2

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

def main(dir="VOC"):
    #[batch,width,height,chanel]
    input_ph = tf.placeholder(tf.float32,[None, IMG_SIZE[0], IMG_SIZE[1], 3])
    y = tf.placeholder(tf.float32,[None, IMG_SIZE[0], IMG_SIZE[1], 3])

    result = inference(input_ph)
    error = loss(result, y)
    train_step = train(error)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        test_feed = {}
        test_feed[input_ph]= cv2.imread(FLAGS.data_dir+str(109+i)+'.png')/255.0
        test_feed[y] = [cv2.imread(FLAGS.data_dir+str(113)+'.png')/255.0]

        sess.run(init_op)
        for step in range(FLAGS.train_step):
            feed_dict = {}
            target = [random.randint(0,104) for i in range(FLAGS.batch_size)]

            feed_dict[input_ph]=[cv2.imread(FLAGS.data_dir+str(i+j)+'.png')/255.0 for j in target]
            feed_dict[y] = [cv2.imread(FLAGS.data_dir+str(i+4)+'.png')/255.0 for i in target ]

            print("step%d training"%step)
            sess.run(train_step, feed_dict=feed_dict)

            if (step+1) % 10 == 0:
                created, error_val = sess.run([result, error], feed_dict=test_feed)
                print("step%d loss: %f" % (step, error_val))
                cv2.imwrite(FLAGS.out_dir+"step"+str(step)+".png", created[0] * 255.0)



if __name__ == "__main__":
    main()
