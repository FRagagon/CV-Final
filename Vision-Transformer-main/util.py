import numpy as np
import os
import pickle
import tensorflow as tf

"""
get_image
"""
def get_image():
    image_path = 'cifar-100-python'
    class_path = os.path.join(image_path,"meta")
    train_path =  os.path.join(image_path,"train")
    test_path = os.path.join(image_path, "test")

    with open(class_path,"rb") as f:
        dic_class = pickle.load(f, encoding='bytes')
        labels = dic_class[b'fine_label_names']

    with open(train_path,"rb") as f:
        dic_train = pickle.load(f, encoding='bytes')
        # labeled_train = list(zip(dic_train[b'data'], tf.one_hot(indices=dic_train[b'fine_labels'], depth=100, on_value=1.0,
        #                                                        off_value=0.0)))
        labeled_train = list(
            zip(dic_train[b'data'], dic_train[b'fine_labels']))


    rdata = []
    rlabel = []
    for d,l in labeled_train:
        rdata.append(np.reshape(np.reshape(d,[3,1024]).T, 32*32*3))
        rlabel.append(l)

    rdata = tf.constant(np.asarray(rdata), dtype=tf.float32)
    rlabel = tf.constant(np.asarray(rlabel))

    with open(test_path, "rb") as f:
        dic_test = pickle.load(f, encoding="bytes")
        data = tf.constant(np.asarray(dic_test[b'data']), dtype=tf.float32)
        test = []
        for d in data:
            test.append((np.reshape(np.reshape(d, [3,1024]).T, [32,32,3])))
        # labeled_test = list(zip(test, tf.one_hot(indices=dic_test[b'fine_labels'], depth=100, on_value=1.0,
        #                                                         off_value=0.0)))
        labeled_test = list(zip(test, dic_test[b'fine_labels']))
    vdata = []
    vlabel = []
    for d,l in labeled_test:
        vdata.append(d)
        vlabel.append(l)

    vdata = tf.constant(np.asarray(vdata), dtype=tf.float32)
    vlabel = tf.constant(np.asarray(vlabel))
    validation = (vdata, vlabel)
    return rdata, rlabel, vdata, vlabel

