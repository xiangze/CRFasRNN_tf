import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
#from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging

class CRFasRNNCelll(tf.nn.rnn_cell.RNNCell):
    def __init__(self.input_size=None, 
                imsize=[128, 128,3], kernsize=[2,2] ,stride=[1,5,5,1], 
                 labelnum=3,
                 cell_clip=None,  initializer=None, state_is_tuple=False):
        self.imsize=imsize
        self.kernsize=kernsize
        self.labelnum=labelnum
        self.stride=stride

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        #I input image (3D)
        #U unary function (pixeliwise label) U_i(l)=-phi(X_i=l) (3D)
        I,U=array_ops.split(3, 2, inputs)
        Q=state #3D 

        with tf.variable_scope("crfasrnn_cell", 
                               initializer = None) as  scope:
            kern=tf.get_variable("kern",
                                 [self.kernel_size[0], self.kernel_size[1], total_arg_size, output_size])

            weight=tf.get_variable("w",[self.kernel_size[0]])
            mu=tf.get_variable("mu",[self.labelnum,self.labelnum])

            Q0 = tf.nn.conv2d(Q,kern, self.stride, padding='SAME') #m,i
            Q1 = tf.reduce_sum(tf.matmul(w,Q0)) #i
            Q2 = tf.redulce_sum(tf.matmul(mu,Q1)
            Z=tf.reduce_sum(Q2)
            Qout= tf.div(Q,Z)

        nstate=array_ops.concat(3, [c, m]))
        return Qout, Qout
