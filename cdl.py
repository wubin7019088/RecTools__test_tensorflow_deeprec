# Collaborative Deep Learning for Recommender Systems
import tensorflow as tf
import basic_nn
import sys

#class CDL(BasicNN):
#    def __init__(self):
#        self.name = 'CDL'
#        self.item_autoencoder_input_dim=100
#        self.item_autoencoder_hidden_dims = [50, 40, 50]
#        self.item_autoencoder_output_dim=100    
        
         
def inference(user_batch=None, usercontent_batch=None
                , item_batch=None, itemcontent_batch=None
                , user_num = 100, item_num = 100, dim = 50
                , item_autoencoder_input_dim = 100, item_autoencoder_hidden_dims = [50, 40, 50]
                , user_autoencoder_input_dim = 100, user_autoencoder_hidden_dims = [50, 40, 50]
                , is_train=True, device="/gpu:1"):
    
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.device(device):
        if itemcontent_batch is not None:
            item_layers = []
            item_layers.append(itemcontent_batch)
            with tf.variable_scope("item_input"):
                item_layer = basic_nn.full_connect(item_layers[0], [item_autoencoder_input_dim, item_autoencoder_hidden_dims[0]],
                                        [item_autoencoder_hidden_dims[0]], is_train=is_train)
                item_layers.append(item_layer)
            for i in range(1, len(item_autoencoder_hidden_dims)):
                with tf.variable_scope("item_layer" + str(i)):
                    item_layer = basic_nn.full_connect(item_layers[-1], [item_autoencoder_hidden_dims[i-1], item_autoencoder_hidden_dims[i]],
                                            [item_autoencoder_hidden_dims[i]], is_train=is_train)
                    item_layers.append(item_layer)
            with tf.variable_scope("item_output"):
                item_layer = basic_nn.full_connect(item_layers[-1], [item_autoencoder_hidden_dims[len(item_autoencoder_hidden_dims)-1], item_autoencoder_input_dim],
                                        [item_autoencoder_input_dim], is_train=is_train)
                item_layers.append(item_layer)

        if usercontent_batch is not None:
            user_layers = []
            user_layers.append(usercontent_batch)
            with tf.variable_scope("user_input"):
                user_layer = basic_nn.full_connect(user_layers[0], [user_autoencoder_input_dim, user_autoencoder_hidden_dims[0]],
                                        [user_autoencoder_hidden_dims[0]], is_train=is_train)
                user_layers.append(user_layer)
            for i in range(1, len(user_autoencoder_hidden_dims)):
                with tf.variable_scope("user_layer" + str(i)):
                    user_layer = basic_nn.full_connect(user_layers[-1], [user_autoencoder_hidden_dims[i-1], user_autoencoder_hidden_dims[i]],
                                            [user_autoencoder_hidden_dims[i]], is_train=is_train)
                    user_layers.append(user_layer)
            with tf.variable_scope("user_output"):
                user_layer = basic_nn.full_connect(user_layers[-1], [user_autoencoder_hidden_dims[len(user_autoencoder_hidden_dims)-1], user_autoencoder_input_dim],
                                        [user_autoencoder_input_dim], is_train=is_train)
                user_layers.append(user_layer)
        if user_batch is None and usercontent_batch is None:
            print('exit')
            sys.exit(0)
        elif user_batch is None:
            user_combine = user_layers[len(user_layers)//2]
        elif usercontent_batch is None:
            user_combine = embd_user
        else:
            user_combine = tf.add(embd_user, user_layers[len(user_layers)//2]) 

        if item_batch is None and itemcontent_batch is None:
            print('exit')
            sys.exit(0)
        elif item_batch is None:
            item_combine = item_layers[len(item_layers)//2]
        elif itemcontent_batch is None:
            item_combine = embd_item
        else:
            item_combine = tf.add(embd_item, item_layers[len(item_layers)//2]) 
                           
        infer = tf.reduce_sum(tf.multiply(user_combine, item_combine), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="cdl_inference")
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item))
    return infer, regularizer
        


            
     