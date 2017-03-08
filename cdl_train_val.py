import time
from collections import deque

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

import dataio
import ops
import cdl

np.random.seed(13575)

BATCH_SIZE = 1000
USER_NUM = 6040
ITEM_NUM = 3952
DIM = 15
EPOCH_MAX = 10
ITEM_CONTENT_DIM = 100
USER_CONTENT_DIM= 100
DEVICE = "/cpu:0"


def clip(x):
    return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def get_data():
    df = dataio.read_process(r"D:\Users\fuzzhang\software\tensorflow\TF_Recommend_Basic\TF_Recommend_Basic\TF-recomm\ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test


def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    item_content_batch = tf.placeholder(tf.float32, shape=[None,ITEM_CONTENT_DIM], name="content_item")
    user_content_batch = tf.placeholder(tf.float32, shape=[None,USER_CONTENT_DIM], name="content_user")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = cdl.inference(user_batch,user_content_batch,item_batch,item_content_batch,user_num=USER_NUM
                                       ,item_num=ITEM_NUM,dim=DIM
                                       ,item_autoencoder_input_dim=ITEM_CONTENT_DIM,item_autoencoder_hidden_dims=[50,DIM,50]
                                       ,user_autoencoder_input_dim=USER_CONTENT_DIM,user_autoencoder_hidden_dims=[30,DIM,30]
                                       ,device="/gpu:0")
    #infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
    #                                       device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            items_content = np.random.randn(BATCH_SIZE,ITEM_CONTENT_DIM).astype(np.float32)
            user_content = np.random.randn(BATCH_SIZE,USER_CONTENT_DIM).astype(np.float32)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   item_content_batch: items_content,
                                                                   user_content_batch: user_content
                                                                   })
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    items_content = np.random.randn(len(users),ITEM_CONTENT_DIM).astype(np.float32)
                    user_content = np.random.randn(len(users),USER_CONTENT_DIM).astype(np.float32)
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,
                                                            item_content_batch: items_content,
                                                            user_content_batch: user_content
                                                            })
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end


if __name__ == '__main__':
    df_train, df_test = get_data()
    svd(df_train, df_test)
    print("Done!")
