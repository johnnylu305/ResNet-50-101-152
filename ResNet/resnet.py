import tensorflow as tf


UNITS = {'resnet_v2_50': [3, 4, 6, 3], 'resnet_v2_101': [3, 4, 23, 3],
         'resnet_v2_152': [3, 8, 36, 3]}
CHANNELS = [64, 128, 256, 512]


def bottleneck(net, channel, is_train, holes=1, c_name='pretrain', stride=1,
               shortcut_conv=False, key=tf.GraphKeys.GLOBAL_VARIABLES):
    with tf.variable_scope('bottleneck_v2', reuse=tf.AUTO_REUSE):
        # define initializer for weights and biases
        w_initializer = tf.contrib.layers.xavier_initializer()
        b_initializer = tf.zeros_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        # batch normalization
        net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                            training=is_train, name='preact')
        net = tf.nn.relu(net)

        # shortcut
        if shortcut_conv:
            with tf.variable_scope('shortcut', reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(initializer=w_initializer,
                                         shape=[1, 1, net.shape[-1],
                                                channel*4],
                                         name='weights',
                                         regularizer=regularizer,
                                         collections=['pretrain', key])
                # convolution for shortcut in order to output size
                shortcut = tf.nn.conv2d(input=net, filter=kernel,
                                        strides=[1, stride, stride, 1],
                                        padding='SAME')
                biases = tf.get_variable(initializer=b_initializer,
                                         shape=channel*4, name='biases',
                                         regularizer=regularizer,
                                         collections=['pretrain', key])
                shortcut = tf.nn.bias_add(shortcut, biases)
        else:
            # shortcut
            shortcut = net

        # convolution 1
        with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(initializer=w_initializer,
                                     shape=[1, 1, net.shape[-1], channel],
                                     name='weights', regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.atrous_conv2d(value=net, filters=kernel, rate=holes,
                                      padding='SAME')
            biases = tf.get_variable(initializer=b_initializer,
                                     shape=channel, name='biases',
                                     regularizer=regularizer,
                                     collections=['non_pretrain', key])
            net = tf.nn.bias_add(net, biases)
            # batch normalization
            net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                                training=is_train,
                                                name='preact')
            net = tf.nn.relu(net)

        # convolution 2
        with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(initializer=w_initializer,
                                     shape=[3, 3, channel, channel],
                                     name='weights', regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.conv2d(input=net, filter=kernel,
                               strides=[1, stride, stride, 1], padding='SAME')
            biases = tf.get_variable(initializer=b_initializer,
                                     shape=channel, name='biases',
                                     regularizer=regularizer,
                                     collections=['non_pretrain', key])
            net = tf.nn.bias_add(net, biases)
            # batch normalization
            net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                                training=is_train,
                                                name='preact')
            net = tf.nn.relu(net)

        # convolution 3
        with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(initializer=w_initializer,
                                     shape=[1, 1, channel, channel*4],
                                     name='weights', regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.atrous_conv2d(value=net, filters=kernel, rate=holes,
                                      padding='SAME')
            biases = tf.get_variable(initializer=b_initializer,
                                     shape=channel*4, name='biases',
                                     regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.bias_add(net, biases)

    return net, shortcut


def block(net, name, unit, channel, is_train):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(unit):
            with tf.variable_scope('unit_'+str(i+1), reuse=tf.AUTO_REUSE):
                # block1 i=0 stride=1
                if i == 0:
                    if name != 'block1':
                        net, shortcut = bottleneck(net, channel, is_train,
                                                   stride=2,
                                                   shortcut_conv=True)
                    else:
                        net, shortcut = bottleneck(net, channel, is_train,
                                                   stride=1,
                                                   shortcut_conv=True)
                else:
                    net, shortcut = bottleneck(net, channel, is_train)
            net = tf.add(net, shortcut)

    return net


def resnet(input_, resnet_v2, is_train, classes):
    key = tf.GraphKeys.GLOBAL_VARIABLES
    with tf.variable_scope(resnet_v2, reuse=tf.AUTO_REUSE):
        # define initializer for weights and biases
        w_initializer = tf.contrib.layers.xavier_initializer()
        b_initializer = tf.zeros_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        # convolution 1
        with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(initializer=w_initializer,
                                     shape=[7, 7, 3, 64],
                                     name='weights', regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.conv2d(input=input_, filter=kernel,
                               strides=[1, 2, 2, 1], padding='SAME')
            biases = tf.get_variable(initializer=b_initializer, shape=64,
                                     name='biases', regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.bias_add(net, biases)
            net = tf.nn.max_pool(value=net, ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')

        for i in range(4):
            net = block(net, 'block'+str(i+1), UNITS[resnet_v2][i],
                        CHANNELS[i], is_train)

        net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                            training=is_train, name='postnorm')
        net = tf.nn.relu(net)

        h, w = net.shape[1:3]
        net = tf.nn.avg_pool(value=net, ksize=[1, h, w, 1],
                             strides=[1, 1, 1, 1], padding='VALID')

    # logits is not in scope 'resnet_v2' in order to fine-tune
    with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(initializer=w_initializer,
                                 shape=[1, 1, 2048, classes], name='weights',
                                 regularizer=regularizer,
                                 collections=['non_pretrain', key])
        net = tf.nn.conv2d(input=net, filter=kernel,
                           strides=[1, 1, 1, 1], padding='VALID')
        biases = tf.get_variable(initializer=b_initializer, shape=classes,
                                 name='biases', regularizer=regularizer,
                                 collections=['non_pretrain', key])
        net = tf.nn.bias_add(net, biases)
    return net
