import numpy as np


SIZE = None
BATCH_SIZE = None


def compute_accuracy(xp, batch_size, is_train, x, y, prediction, sess):
    match_ = 0.0
    global SIZE, BATCH_SIZE
    SIZE = np.size(y)
    BATCH_SIZE = batch_size
    ix, iter_ = to_batch(False)
    for i in range(iter_):
        # run prediction
        prediction_ = sess.run(prediction,
                               feed_dict={xp: x[ix[i]],
                                          is_train: False})
        # compute accuracy
        match = np.sum(y[ix[i]] == prediction_)
        match_ = match_ + match
    accuracy_ = match_/SIZE
    # print accuracy
    print('accuracy : {}'.format(accuracy_))
    return accuracy_


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
