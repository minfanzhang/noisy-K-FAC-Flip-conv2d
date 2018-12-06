import tensorflow as tf


def _append_homog(tensor):
    rank = len(tensor.shape.as_list())
    shape = tf.concat([tf.shape(tensor)[:-1], [1]], axis=0)
    ones = tf.ones(shape, dtype=tensor.dtype)
    return tf.concat([tensor, ones], axis=rank - 1)


def dense(inputs, weights, batch_norm, is_training, particles=1):
    inputs = _append_homog(inputs)
    n_in = inputs.shape.as_list()[-1]
    inputs = tf.reshape(inputs, [particles, -1, n_in])
    preactivations = tf.matmul(inputs, weights)
    preactivations = tf.reshape(preactivations, [-1, weights.get_shape()[-1]])

    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)
    print("activation in dense has shape : ", activations.shape.as_list())
    return preactivations, activations

def flipoutlayerFC(x, W_0, u_c, v_c, is_training, batch_norm, particles=1, use_flipout=False):
    #W_0 = tf.tile(tf.expand_dims(W_0, 0), [particles, 1, 1])
    weight_dim = W_0.shape.as_list()

    x = _append_homog(x)
    u_c = tf.tile(tf.expand_dims(u_c, 0), [particles, 1, 1])
    v_c = tf.tile(tf.expand_dims(v_c, 0), [particles, 1, 1])
    n_in = x.shape.as_list()[-1]
    x = tf.reshape(x, [particles, -1, n_in])
    out_W_0 = tf.tile(tf.expand_dims(W_0, 0), [particles, 1, 1])
    def flipout_multi():
        # x is n*m where m is the dimension, n is the mini-batch size.
        # W_0 is m*h where h is the num of hidden units.
        epsilon = tf.truncated_normal(weight_dim, stddev=1.0)
        # delta_W = tf.multiply(W_0, epsilon)
        delta_W = epsilon
        if use_flipout:
            def generate_flipping_factor(dim):
                shape = tf.stack([tf.shape(x)[0], dim])
                random = tf.truncated_normal(shape)
                positives = tf.ones(shape)
                negatives = tf.zeros(shape)-1
                return tf.where(random>0, positives, negatives)
            E1 = generate_flipping_factor(weight_dim[1])
            E2 = generate_flipping_factor(weight_dim[0])
            pert_x = tf.matmul(tf.multiply(tf.matmul(tf.multiply(tf.matmul(x, u_c), E2), delta_W), E1), v_c)
            return (tf.matmul(x, W_0) + pert_x)
        else:
            rand = tf.random_normal(shape=tf.shape(out_W_0))
            #u_c = tf.tile(tf.expand_dims(u_c, 0), [particles, 1, 1])
            #v_c = tf.tile(tf.expand_dims(v_c, 0), [particles, 1, 1])
            out_rand = tf.matmul(u_c, tf.matmul(rand, v_c, transpose_b=True))
            W_0_final = out_W_0 + out_rand

            #n_in = x.shape.as_list()[-1]
            #x = tf.reshape(x, [particles, -1, n_in])

            preactivations = tf.matmul(x, W_0_final)
            preactivations = tf.reshape(preactivations, [-1, W_0_final.get_shape()[-1]])

            return preactivations

    def FCLayer():
        # Used during inference.
        return tf.matmul(x, out_W_0)

    preactivations = tf.cond(is_training, lambda: flipout_multi(), lambda: FCLayer())
    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)
    return preactivations, activations

def test_FC(inputs, weights, u_c, v_c, is_training, batch_norm, particles=1):
    weights = tf.tile(tf.expand_dims(weights, 0), [particles, 1, 1])
    rand = tf.random_normal(shape=tf.shape(weights))

    u_c = tf.tile(tf.expand_dims(u_c, 0), [particles, 1, 1])
    v_c = tf.tile(tf.expand_dims(v_c, 0), [particles, 1, 1])

    inputs = _append_homog(inputs)
    n_in = inputs.shape.as_list()[-1]
    inputs = tf.reshape(inputs, [particles, -1, n_in])

    out_rand = tf.matmul(u_c, tf.matmul(rand, v_c, transpose_b=True))
    weights = weights + out_rand

    preactivations = tf.matmul(inputs, weights)
    preactivations = tf.reshape(preactivations, [-1, weights.get_shape()[-1]])

    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)
    return preactivations, activations

def test_FC_flip(inputs, weights, u_c, v_c, is_training, batch_norm, particles=1):
    weights = tf.tile(tf.expand_dims(weights, 0), [particles, 1, 1])
    rand = tf.random_normal(shape=tf.shape(weights))

    u_c = tf.tile(tf.expand_dims(u_c, 0), [particles, 1, 1])
    v_c = tf.tile(tf.expand_dims(v_c, 0), [particles, 1, 1])

    def generate_flipping_factor(dim):
        shape = tf.stack([tf.shape(inputs)[0], dim])
        random = tf.truncated_normal(shape)
        positives = tf.ones(shape)
        negatives = tf.zeros(shape)-1
        return tf.where(random>0, positives, negatives)
    E1 = generate_flipping_factor(weights.shape.as_list()[2])
    E2 = generate_flipping_factor(weights.shape.as_list()[1])
    E1 = tf.tile(tf.expand_dims(E1, 0), [particles, 1, 1])
    E2 = tf.tile(tf.expand_dims(E2, 0), [particles, 1, 1])


    inputs = _append_homog(inputs)
    n_in = inputs.shape.as_list()[-1]
    inputs = tf.reshape(inputs, [particles, -1, n_in])

    #out_rand = tf.matmul(u_c, tf.matmul(rand, v_c, transpose_b=True))
    #weights = weights + out_rand

    out_x = tf.matmul(inputs, weights)
    pert_x = tf.matmul(tf.multiply(tf.matmul(tf.multiply(tf.matmul(inputs, u_c), E2), rand), E1), v_c)
    preactivations = out_x + pert_x

    #preactivations = tf.matmul(inputs, weights)
    preactivations = tf.reshape(preactivations, [-1, weights.get_shape()[-1]])
    
    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)
    return preactivations, activations

def conv2d(inputs, weights, filter_shape, batch_norm, is_training,
           particles=1, strides=(1, 1, 1, 1), padding="SAME"):
    filter_height, filter_width, in_channels, out_channels = filter_shape

    def f1():
        weight = tf.reshape(weights[0, :-1, :], filter_shape)
        bias = tf.squeeze(weights[0, -1, :])
        conv = tf.nn.conv2d(inputs, filter=weight,
                            strides=strides, padding=padding)
        return tf.nn.bias_add(conv, bias)

    def f2():
        patches = tf.extract_image_patches(
            inputs,
            ksizes=[1, filter_height, filter_width, 1],
            strides=strides,
            rates=[1, 1, 1, 1],
            padding=padding)

        patches = _append_homog(patches)
        pb, h_out, w_out, flatten_size = patches.shape.as_list()
        patches = tf.reshape(patches, [particles, -1, flatten_size])
        preactivations = tf.matmul(patches, weights)
        preactivations = tf.reshape(preactivations, [-1, h_out, w_out, out_channels])
        return preactivations

    preactivations = tf.cond(tf.equal(particles, 1), f1, f2)
    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)
    return preactivations, activations

def test_conv2d_flip(inputs, weights, filter_shape, batch_norm, is_training,
           weights_mean, u_c, v_c, batch_size,
           particles=1, strides=(1, 1, 1, 1), padding="SAME"):
    filter_height, filter_width, in_channels, out_channels = filter_shape
    H = inputs.shape.as_list()[1]
    W = inputs.shape.as_list()[2]

    def f1():
        weight_mean = weights_mean[:-1, :]
        bias_mean = weights_mean[-1, :]
        extracted_patches = tf.extract_image_patches(images=inputs, ksizes=[1, filter_height, filter_width, 1], 
                                                     strides=[1,1,1,1], rates=[1, 1, 1, 1], padding=padding)
        def generate_flipping_factor(dim):
            shape = tf.stack([tf.shape(extracted_patches)[0], dim])
            random = tf.truncated_normal(shape)
            positives = tf.ones(shape)
            negatives = tf.zeros(shape)-1
            return tf.where(random>0, positives, negatives)

        E1 = generate_flipping_factor(weight_mean.shape.as_list()[1])
        E2 = generate_flipping_factor(weight_mean.shape.as_list()[0])
        rand = tf.random_normal(shape=tf.shape(weight_mean))

        out1 = tf.tensordot(extracted_patches, u_c, axes=[[3], [0]])
        out2 = tf.transpose(out1, [1,2,0,3])
        out3 = tf.transpose(tf.multiply(out2, E2), [2,0,1,3])
        out4 = tf.tensordot(out3, rand, axes=[[3], [0]])
        out5 = tf.transpose(out4, [1,2,0,3])
        out6 = tf.transpose(tf.multiply(out5, E1), [2,0,1,3])
        out7 = tf.tensordot(out6, v_c, axes=[[3], [0]])


        #weight = tf.reshape(weights[0, :-1, :], filter_shape)
        bias = tf.squeeze(weights[0, -1, :])
        weight_mean_reshaped_to_filter = tf.reshape(weight_mean, filter_shape)
        conv = tf.nn.conv2d(inputs, filter=weight_mean_reshaped_to_filter,
                            strides=strides, padding=padding) + out7

        return tf.nn.bias_add(conv, bias)

    def f2():
        patches = tf.extract_image_patches(
            inputs,
            ksizes=[1, filter_height, filter_width, 1],
            strides=strides,
            rates=[1, 1, 1, 1],
            padding=padding)

        patches = _append_homog(patches)
        pb, h_out, w_out, flatten_size = patches.shape.as_list()
        patches = tf.reshape(patches, [particles, -1, flatten_size])
        preactivations = tf.matmul(patches, weights)
        preactivations = tf.reshape(preactivations, [-1, h_out, w_out, out_channels])
        return preactivations

    preactivations = tf.cond(tf.equal(particles, 1), f1, f2)

    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)
    return preactivations, activations

def generate_flip(weight_tile) :
    random = tf.truncated_normal(shape=weight_tile.shape)
    positives = tf.ones(shape=weight_tile.shape)
    negatives = tf.zeros(shape=weight_tile.shape)-1
    flip = tf.where(random>0, positives, negatives)
    return flip