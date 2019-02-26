import tensorflow as tf
import numpy as np
import os
import sys
import resnet
import parser
import load_data


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
WIDTH = 224
HEIGHT = 224


KEY = tf.GraphKeys.GLOBAL_VARIABLES
# class
class_ = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed",
          "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize",
          "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
          "Sugar beet"]


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


def test_net(x, n):
    # set placeholder
    xp = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 3))
    is_train = tf.placeholder(tf.bool)
    # get network
    prediction = net_(xp, is_train)
    with tf.Session() as sess:
        with open(BASEDIR+"/dataset/" + RESNET_V2 +
                  "_submission.csv", 'w') as sub:
            np.savetxt(sub, [['file', 'species']],
                       fmt="%s,%s")
            # setup saver
            restorer = tf.train.Saver(tf.global_variables())
            # load weight
            restorer.restore(sess, RESTORE_CKPT_PATH)
            ix, iter_ = to_batch(False)
            for i in range(iter_):
                # run prediction
                prediction_ = sess.run(prediction,
                                       feed_dict={xp: x[ix[i]],
                                                  is_train: False})
                batch_size_ = np.size(ix[i])
                for j in range(batch_size_):
                    # save as .csv file
                    np.savetxt(sub, [[n[ix[i][j]], class_[prediction_[j]]]],
                               fmt="%s,%s")


def to_batch(pad=False):
    if pad or SIZE % BATCH_SIZE == 0:
        pad_size = SIZE % BATCH_SIZE
        ix = np.random.permutation(SIZE)
        ix = np.append(ix, np.random.choice(ix, pad_size))
        iter_ = int((SIZE + pad_size)/BATCH_SIZE)
        ix = np.array_split(ix, iter_)
    else:
        ix = np.random.permutation(SIZE)
        iter_ = int(SIZE/BATCH_SIZE) + 1
        ix = np.split(ix, [x*BATCH_SIZE for x in range(1, iter_)])
    return ix, iter_


def main():
    # get data
    x_test, x_name = load_data.load('test_resize', label=False)
    global SIZE
    SIZE = np.size(x_name)
    ix, iter_ = to_batch(pad=False)
    # train network
    test_net(x_test, x_name)


if __name__ == '__main__':
    main()
