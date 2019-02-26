import tensorflow as tf
import numpy as np
import os
import sys
import math
import resnet
import parser
import load_data
import eval_


BASEDIR = os.path.join(os.path.dirname(__file__), './')


# get argument
args = parser.test_parser()

# standard output format
SPACE = 35

# default: resnet_v2_101
RESNET_V2 = 'resnet_v2_' + args.layers
# default: 6
CLASSES = args.classes
# default: 16
BATCH_SIZE = args.batch
# defalut: -1
RESTORE_TARGET = args.recover
# restore weights path
RESTORE_CKPT_PATH = BASEDIR + "/models/" + RESNET_V2 + "/model_" +\
                    str(RESTORE_TARGET) + ".ckpt"

if not os.path.isfile(RESTORE_CKPT_PATH + ".index"):
    print("Recover target not found.")
    sys.exit()
SIZE = None
ITER = None
WIDTH = 224
HEIGHT = 224


KEY = tf.GraphKeys.GLOBAL_VARIABLES


# crop center 224*224
def crop_center(img):
    img_ = []
    size_ = img.shape[0]
    for i in range(size_):
        h, w = img[i].shape[0:2]
        # random crop
        shift1 = int((h-HEIGHT)/2)
        shift2 = int((w-WIDTH)/2)
        img_.append(img[i][shift1:HEIGHT+shift1, shift2:WIDTH+shift2][:])
    return np.asarray(img_)


def net_(xp, is_train):
    x = xp
    # create network
    net = resnet.resnet(x, RESNET_V2, is_train, CLASSES)
    # squeeze
    net = tf.squeeze(net, axis=(1, 2))

    prediction = tf.argmax(net, axis=1)

    return prediction


def val_net(x_val, y_val):
    # set placeholder
    xp = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 3))
    is_train = tf.placeholder(tf.bool)
    # get network
    prediction = net_(xp, is_train)
    with tf.Session() as sess:
        # setup saver
        restorer = tf.train.Saver(tf.global_variables())
        # load weight
        restorer.restore(sess, RESTORE_CKPT_PATH)
        print('Val acc:')
        eval_.compute_accuracy(xp, BATCH_SIZE, is_train, x_val,
                               y_val, prediction, sess)


def main():
    # get data
    x_val, y_val = load_data.load('val_resize')
    global SIZE
    SIZE = np.size(y_val)
    global ITER
    ITER = int(math.ceil(SIZE/BATCH_SIZE))
    # train network
    val_net(x_val, y_val)


if __name__ == '__main__':
    main()
