import tensorflow as tf
import numpy as np

from misc.layers import dense, conv2d, flipoutlayerFC, test_FC, test_FC_flip, test_conv2d_flip
from network.registry import register_model


def VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, num_blocks, config):
    def VGGBlock(inputs, layers, out_channel, layer_idx, use_flip, batch_size):
        l2_loss = 0.
        for l in range(layers):
            in_channel = inputs.shape.as_list()[-1]
            sampler.register_block(layer_idx+l, (3, 3, in_channel, out_channel))
            if use_flip :
                weights = sampler.sample(layer_idx+l)
                weights_mean, u_c, v_c = sampler.sample_mean_and_var(layer_idx+l, nobias_flag=True)
                l2_loss += 0.5 * tf.reduce_sum(weights_mean ** 2)
                pre, act = test_conv2d_flip(inputs, weights, (3, 3, in_channel, out_channel), batch_norm, is_training, weights_mean, u_c, v_c, config.batch_size, particles, padding="SAME")

            else :
                weights = sampler.sample(layer_idx+l)
                l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
                pre, act = conv2d(inputs, weights, (3, 3, in_channel, out_channel),
                                  batch_norm, is_training, particles, padding="SAME")
            layer_collection.register_conv2d(sampler.get_params(layer_idx+l), (1, 1, 1, 1), "SAME", inputs, pre)
            inputs = act
            
        outputs = tf.layers.max_pooling2d(inputs, 2, 2, "SAME")
        return outputs, l2_loss

    def VGG_FC_Block(inputs, layers, input_size, output_size, layer_idx, use_flip) :
        inputs = tf.reshape(inputs, shape=[-1, int(np.prod(inputs.shape[1:]))])
        l2_loss = 0.
        for l in range(layers):
            if l == 0 :
                sampler.register_block(layer_idx+l, (input_size, output_size))
            else :
                sampler.register_block(layer_idx+l, (output_size, output_size))

            if use_flip :
                weights_mean, u_c, v_c = sampler.sample_mean_and_var(layer_idx+l)
                l2_loss += 0.5 * tf.reduce_sum(weights_mean ** 2)
                _, outputs = test_FC_flip(inputs, weights_mean, u_c, v_c, is_training, batch_norm, particles)
            else :
                weights = sampler.sample(layer_idx+l)
                l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
                _, outputs = dense(inputs, weights, batch_norm, is_training, particles)
            #_, outputs = flipoutlayerFC(inputs, weights_mean, u_c, v_c, is_training, batch_norm, particles)
            layer_collection.register_fully_connected(sampler.get_params(layer_idx+l), inputs, outputs)
            inputs = outputs
        outputs = inputs
        return outputs, l2_loss

    if config.use_conv2d :
        inputs = tf.tile(inputs, [particles, 1, 1, 1])

        num_blocks_conv2d = [2, 2, 3, 3, 3]

        layer_idx = 0
        # block 1
        layer1, l2_loss1 = VGGBlock(inputs, num_blocks_conv2d[0], 32, layer_idx, config.use_flip, config.batch_size)
        layer_idx += num_blocks_conv2d[0]
        # block 2
        layer2, l2_loss2 = VGGBlock(layer1, num_blocks_conv2d[1], 64, layer_idx, config.use_flip, config.batch_size)
        layer_idx += num_blocks_conv2d[1]
        # block 3
        layer3, l2_loss3 = VGGBlock(layer2, num_blocks_conv2d[2], 128, layer_idx, config.use_flip, config.batch_size)
        layer_idx += num_blocks_conv2d[2]
        # block 4
        layer4, l2_loss4 = VGGBlock(layer3, num_blocks_conv2d[3], 256, layer_idx, config.use_flip, config.batch_size)
        layer_idx += num_blocks_conv2d[3]
        # block 5
        layer5, l2_loss5 = VGGBlock(layer4, num_blocks_conv2d[4], 256, layer_idx, config.use_flip, config.batch_size)
        layer_idx += num_blocks_conv2d[4]

        l2_loss = l2_loss1 + l2_loss2 + l2_loss3 + l2_loss4 + l2_loss5

        flat = tf.reshape(layer5, shape=[-1, int(np.prod(layer5.shape[1:]))])
        sampler.register_block(layer_idx, (256, 10))
        weights = sampler.sample(layer_idx)
        l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
        logits, _ = dense(flat, weights, batch_norm, is_training, particles)
        layer_collection.register_fully_connected(sampler.get_params(layer_idx), flat, logits)
        layer_collection.register_categorical_predictive_distribution(logits, name="logits")

        return logits, l2_loss

    else :
        print("inputs_origin has shape : ", inputs.shape.as_list())
        input_flat_shape = 0
        if sampler.config.dataset == 'mnist' :
            inputs = tf.tile(inputs, [particles, 1])
            inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])
            print("inputs has shape : ", inputs.shape.as_list(), " after reshape to image")
            input_flat_shape = 784
        else :
            inputs = tf.tile(inputs, [particles, 1, 1, 1])
            input_flat_shape = 3072
        layer_idx = 0
        # block 1
        #layer1, l2_loss1 = VGGBlock(inputs, num_blocks[0], 32, layer_idx)
        layer1, l2_loss1 = VGG_FC_Block(inputs, num_blocks[0], input_flat_shape, 512, layer_idx, config.use_flip)
        layer_idx += num_blocks[0]
        # block 2
        #layer2, l2_loss2 = VGGBlock(layer1, num_blocks[1], 64, layer_idx)
        layer2, l2_loss2 = VGG_FC_Block(layer1, num_blocks[1], 512, 512, layer_idx, config.use_flip)
        layer_idx += num_blocks[1]
        # block 3
        #layer3, l2_loss3 = VGGBlock(layer2, num_blocks[2], 128, layer_idx)
        layer3, l2_loss3 = VGG_FC_Block(layer2, num_blocks[2], 512, 512, layer_idx, config.use_flip)
        layer_idx += num_blocks[2]
        # block 4
        #layer4, l2_loss4 = VGGBlock(layer3, num_blocks[3], 256, layer_idx)
        #layer4, l2_loss4 = VGG_FC_Block(layer3, num_blocks[3], 512, 512, layer_idx, config.use_flip)
        #layer_idx += num_blocks[3]
        # block 5
        #layer5, l2_loss5 = VGGBlock(layer4, num_blocks[4], 256, layer_idx)
        #layer5, l2_loss5 = VGG_FC_Block(layer2, 1, 512, 256, layer_idx, config.use_flip)
        #layer_idx += 1


        # l2_loss
        l2_loss = l2_loss1 + l2_loss2 + l2_loss3

        flat = tf.reshape(layer3, shape=[-1, int(np.prod(layer3.shape[1:]))])

        #layer6, l2_loss6 = VGG_FC_Block(flat, 1, 256, 256, layer_idx, config.use_flip)
        #l2_loss += l2_loss6
        #layer_idx += 1

        #flat = layer6

        sampler.register_block(layer_idx, (512, 10))
        weights = sampler.sample(layer_idx)
        l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
        logits, _ = dense(flat, weights, batch_norm, is_training, particles)
        layer_collection.register_fully_connected(sampler.get_params(layer_idx), flat, logits)
        layer_collection.register_categorical_predictive_distribution(logits, name="logits")

        return logits, l2_loss


@register_model("vgg11")
def VGG11(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [1, 1, 2, 2, 2], config)


@register_model("vgg13")
def VGG13(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 2, 2, 2], config)


@register_model("vgg16")
def VGG16(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 1, 1, 1, 3], config)


@register_model("vgg19")
def VGG19(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 4, 4, 4], config)
