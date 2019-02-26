import tensorflow as tf
import random
import numpy as np
import os
import sys
import resnet
import parser
import load_data
import eval_

BASEDIR = os.path.join(os.path.dirname(__file__), './')


# get argument
args = parser.train_parser()

# standard output format
SPACE = 35

# default: resnet_v2_101
RESNET_V2 = 'resnet_v2_' + args.layers
# default: ./pretrain_models/resnet_v2_101.ckpt'
RESNET_V2_CKPT_PATH = BASEDIR+'/pretrain_models/'+RESNET_V2+'.ckpt'
# default: 12
CLASSES = args.classes
# default: 16
BATCH_SIZE = args.batch
# default: 0.001
LR = args.lr
# default: 300
EPOCH = args.epoch
# default: True
PRETRAIN = args.pretrain
# defalut: 1
SAVE_STEP = args.save
# defalut: -1
RESTORE_TARGET = args.recover
# defalut: False
ADD_VAL = args.val

# restore weights path
RESTORE_CKPT_PATH = BASEDIR + "/models/" + RESNET_V2 + "/model_" +\
                    str(RESTORE_TARGET) + ".ckpt"
if RESTORE_TARGET == -1:
    if not os.path.exists(BASEDIR + "/models/" + RESNET_V2):
            os.makedirs(BASEDIR + "/models/" + RESNET_V2)
elif not os.path.isfile(RESTORE_CKPT_PATH + ".index"):
    print("Recover target not found.")
    sys.exit()
SIZE = None
WIDTH = 224
HEIGHT = 224
# learning decay step
# default: 300
DECAY_STEP = 300
# learning rate decay rate
# default: 0.1
DECAY_RATE = 0.1
# staircase
# default: False
STAIRCASE = False


KEY = tf.GraphKeys.GLOBAL_VARIABLES


# augmentation
def augmentation(img):
    img_ = []
    size_ = img.shape[0]
    for i in range(size_):
        h, w = img[i].shape[0:2]
        # random crop
        shift1 = random.randint(0, h-HEIGHT)
        shift2 = random.randint(0, w-WIDTH)
        img_.append(img[i][shift1:HEIGHT+shift1, shift2:WIDTH+shift2][:])
        # flip
        if random.randint(0, 1) == 0:
            img_[i] = np.flip(img_[i], 1)
    return img_


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


def net_(xp, yp, is_train, global_step):
    x = xp
    # create network
    net = resnet.resnet(x, RESNET_V2, is_train, CLASSES)
    # squeeze
    net = tf.squeeze(net, axis=(1, 2))

    # to one hot
    y = tf.one_hot(yp, depth=CLASSES)

    # define loss
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y))

    lr = tf.train.exponential_decay(LR, global_step, DECAY_STEP, DECAY_RATE,
                                    STAIRCASE)

    # get pretrain variable
    var_pre = tf.get_collection('pretrain')
    # get non-pretrain variable
    var_non = list(set(tf.global_variables()) - set(var_pre))
    # operations for batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope('optimizer'):
        # batch normalization operations added as a dependency
        with tf.control_dependencies(update_ops):
            if PRETRAIN or RESTORE_TARGET != -1:
                # set different learning rate
                opt_pre = tf.train.AdamOptimizer(
                        learning_rate=lr*0.5).minimize(loss, var_list=var_pre)
                opt_non = tf.train.AdamOptimizer(
                        learning_rate=lr).minimize(loss, var_list=var_non)
                opt = tf.group(opt_pre, opt_non)
            else:
                opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    prediction = tf.argmax(net, axis=1)

    return prediction, opt, loss


def train_net(x_train, y_train, x_val, y_val):
    # set placeholder
    xp = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 3))
    yp = tf.placeholder(tf.int32, shape=(None))
    is_train = tf.placeholder(tf.bool)
    global_step = tf.placeholder(tf.int32)
    # get network
    prediction, opt, loss = net_(xp, yp, is_train, global_step)
    best_epoch = 0
    best_acc = 0
    with tf.Session() as sess:
        if PRETRAIN:
            # get pretrain variable
            var_to_restore = tf.get_collection('pretrain')
            # get non-pretrain restore
            var = list(set(tf.global_variables()) - set(var_to_restore))
            # setup restorer
            restorer = tf.train.Saver(var_to_restore)
            # restore weights
            restorer.restore(sess, RESNET_V2_CKPT_PATH)
            # initial non-pretrain variables
            init = tf.variables_initializer(var)
            sess.run(init)
        elif RESTORE_TARGET != -1:
            # setup saver
            restorer = tf.train.Saver()
            # load weight
            restorer.restore(sess, RESTORE_CKPT_PATH)
        else:
            # get variable
            var = tf.global_variables()
            # initial variables
            init = tf.variables_initializer(var)
            sess.run(init)

        # setup saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        for i in range(1+RESTORE_TARGET, EPOCH):
            print('Epoch {}'.format(i))
            ix, iter_ = to_batch(False)
            loss__ = 0
            for j in range(iter_):
                x_train_ = augmentation(x_train[ix[j]])
                opt_, loss_ = sess.run([opt, loss],
                                       feed_dict={xp: x_train_,
                                       yp: y_train[ix[j]],
                                       is_train: True,
                                       global_step: i})
                loss__ = loss__ + loss_*(np.size(ix[j])/iter_)
            print('loss : {}'.format(loss__))
            # test on validation set
            print('Val acc:')
            acc = eval_.compute_accuracy(xp, BATCH_SIZE, is_train, x_val,
                                         y_val, prediction, sess)
            if acc > best_acc:
                best_acc = acc
                best_epoch = i
            if i % SAVE_STEP == 0:
                saver.save(sess, BASEDIR + "/models/" + RESNET_V2 +
                           "/model_" + str(i) + ".ckpt")
    print("Best epoch:", best_epoch)
    print("Best acc:", best_acc)


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
    x_train, y_train = load_data.load('train_resize')
    if ADD_VAL:
        x_val, y_val = load_data.load('val2_resize')
        x_train = np.append(x_train, x_val, axis=0)
        y_train = np.append(y_train, y_val, axis=0)
        x_val = crop_center(x_val)
    else:
        x_val, y_val = load_data.load('val_resize')
    global SIZE
    SIZE = np.size(y_train)
    # train network
    train_net(x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    main()
