import tensorflow as tf



#class BasicNN:
#    def __init__(self):
#        self.name = 'BasicNN' 

def full_connect(inputs, weights_shape, biases_shape, action_fun=None
                 , dropout_keep_prob=1, enable_bn=False, bn_epsilon=0.1, is_train=True
                 ):
    weights = tf.get_variable("weights",
                                weights_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases",
                                biases_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
    layer = tf.matmul(inputs, weights) + biases

    if action_fun == 'relu':
        layer = tf.nn.relu(layer)

    if is_train and dropout_keep_prob < 1:
        layer = tf.nn.dropout(layer, dropout_keep_prob)

    if is_train and enable_bn:
        mean, var = tf.nn.moments(layer, axes=[0])
        scale = tf.get_variable("scale",
                                biases_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
        shift = tf.get_variable("shift",
                                biases_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,bn_epsilon)
    return layer

    

