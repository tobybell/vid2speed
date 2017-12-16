#!/usr/local/opt/python3/bin/python3.6

import numpy as np
import tensorflow as tf

from PIL import Image

# Misc. config.

# Log messages.
tf.logging.set_verbosity(tf.logging.INFO)

# Batch size to use when training. Note that this is the number of image
# *pairs* that will be fed in at once (so it need not be even).
BATCH_SIZE = 12

# Number of vid2speed speed buckets.
NUM_LABELS = 15

# Build the vid2speed classifier network. Accepts a tensor of 3-channel (RGB)
# uint8 image data with shape [2n, 224, 224, 3] - i.e., n pairs of images.
# Returns the logits layer.
def vid2speed(images):

    reshaped = tf.reshape(images, [-1, 2, 224, 224, 3])
    inputs = tf.cast(reshaped, tf.float32)
    
    even_img = inputs[:, 0, :, :, :]
    odd_img = inputs[:, 1, :, :, :]
    tf.summary.image('even', even_img, max_outputs=12)
    tf.summary.image('odd', odd_img, max_outputs=12)

    concat_vol = tf.concat([even_img, odd_img], 3)

    # Three convolutional layers.
    conv_1 = tf.layers.conv2d(concat_vol,
                              filters=64,
                              kernel_size=3,
                              strides=2,
                              activation=tf.nn.relu,
                              name='conv_1')
    conv_2 = tf.layers.conv2d(conv_1,
                              filters=64,
                              kernel_size=3,
                              strides=2,
                              activation=tf.nn.relu,
                              name='conv_2')
    conv_3 = tf.layers.conv2d(conv_2,
                              filters=64,
                              kernel_size=3,
                              strides=2,
                              activation=tf.nn.relu,
                              name='conv_3')

    # Four fully-connected layers.
    flat = tf.reshape(conv_3, [-1, 46656])
    fc_4 = tf.layers.dense(flat, 4096, activation=tf.nn.relu, name='fc_4')
    fc_5 = tf.layers.dense(fc_4, 4096, activation=tf.nn.relu, name='fc_5')
    fc_6 = tf.layers.dense(fc_5, 4096, activation=tf.nn.relu, name='fc_6')
    fc_7 = tf.layers.dense(fc_6, NUM_LABELS, name='fc_7')

    return fc_7

# Train the vid2speed network given an `images` batch and `labels` vector. The
# `images` tensor should have shape [2n, 224, 224, 3] and type uint8 - i.e., a
# batch of n pairs of 3-channel (RGB) 224x224 images. This function is written
# to comply with the TensorFlow Estimator specification, and should be used to
# create an Estimator for the vid2speed network.
def vid2speed_estimator(features, labels, mode):
    
    # Build the network itself.
    images = features['images']
    logits = vid2speed(images)
   
    # Given the logits layer, we can form predictions using argmax to find the
    # most likely class, and softmax to find class probabilities.
    predictions = {
        'classes': tf.cast(tf.argmax(logits, axis=1), tf.int32),
        
        # Adding `softmax_tensor` to the graph is used for predictions. It also
        # allows for logging during training.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    # If we just want to predict new samples, we don't need anything else.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions)
    
    # Construct the loss function based on a one_hot vector for each label and
    # the logits layer from the vid2speed network. The loss function is
    # necessary for training and evaluation.
    labels = labels / 29 * NUM_LABELS
    labels = tf.cast(labels, tf.int32)
    one_hot = tf.one_hot(labels, NUM_LABELS)
    loss = tf.losses.softmax_cross_entropy(one_hot, logits)
    tf.summary.scalar('cost', loss)
    tf.identity(loss, name='loss')
    tf.identity(labels, name='labels')
    tf.identity(predictions['classes'], 'preds')

    # Configure the training operation using the loss function created above.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, tf.train.get_global_step())
        
        correct_prediction = tf.equal(labels, predictions['classes'])
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Add evaluation metrics for EVAL mode.
    metrics = {'accuracy': tf.metrics.accuracy(labels, predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

# Parameter `pair` should have size [2, 224, 224, 3]. Parameter `label` should be a
# single label value.
def get_gradient_map(pair, label):
    tf.reset_default_graph()
    pair = tf.Variable(pair, name='pair', dtype=tf.float32)
    label = np.array([label])
    label = label / 29 * NUM_LABELS
    logits = vid2speed(pair)
    label = tf.cast(label, tf.int32)
    one_hot = tf.one_hot(label, NUM_LABELS)
    loss = tf.losses.softmax_cross_entropy(one_hot, logits)
    with tf.Session() as sess:
        pair.initializer.run()
        variables = tf.trainable_variables()
        uninit = []
        for vari in variables:
            if vari.name != 'pair:0':
                uninit.append(vari)
        saver = tf.train.Saver(uninit)
        saver.restore(sess, './model/model.ckpt-9014')
        grad = tf.gradients(loss, pair)
        return np.array(sess.run(grad))

def main():

    # Data import.

    # Load training and testing data.
    # Note: data tensors should have shape [2n, 224, 224, 3] and type uint8 (i.e.,
    # n pairs of 224x224 3-channel (RGB) images), and the label vectors should have
    # shape [n], containing the speed bucket for each pair of images.
    #train_data = np.load('/data/X_train.npy').reshape((-1, 2, 224, 224, 3))
    #train_labels = np.load('/data/y_train.npy')
    test_data = np.load('/data/X_test.npy').reshape((-1, 2, 224, 224, 3))
    test_labels = np.load('/data/y_test.npy')
    
    normy_index = 17
    pair = test_data[normy_index, :, :, :, :]
    mappy = get_gradient_map(pair, test_labels[normy_index])
    mappy = mappy[0, :, :, :, :]
    mappy_norms = np.linalg.norm(mappy, ord=2, axis=3)
    mappy_norms /= np.max(mappy_norms)
    mappy_norms = (mappy_norms * 255).astype('uint8')
    mappy_norms = np.expand_dims(mappy_norms, axis=3)
    mappy_norms = np.repeat(mappy_norms, 3, axis=3)
    
    im0 = Image.fromarray(pair[0, :, :, :])
    im1 = Image.fromarray(pair[1, :, :, :])
    map0 = Image.fromarray(mappy_norms[0, :, :, :])
    map1 = Image.fromarray(mappy_norms[1, :, :, :])
    im0.save('im0.png')
    im1.save('im1.png')
    map0.save('map0.png')
    map1.save('map1.png')
    
    # Cut off here.
    return

    # Create the Estimator
    config = tf.estimator.RunConfig(save_checkpoints_secs=60)
    v2s_classifier = tf.estimator.Estimator(vid2speed_estimator,
                                            './model',
                                            config=config)

    # Set up logging during training. Logs will be produced every 50 training
    # iterations. Note: alternatively, logs can be produced at a fixed time
    # rate. To enable this behavior, use the `every_n_secs` parameter instead
    # of the `every_n_iter` parameter below.
    log_targets = {
        'loss': 'loss',
        'preds': 'preds',
        'labels': 'labels'
    }
    logging = tf.train.LoggingTensorHook(log_targets, every_n_iter=50)

    '''
    # Create a training function using the training data set above.
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=False
    )

    # Run training.
    v2s_classifier.train(input_fn, steps=20000, hooks=[logging])
    '''
    
    # Create an evaluation function using the test data set above.
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': test_data},
        y=test_labels,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    # Run training.
    v2s_classifier.evaluate(input_fn, hooks=[logging])
    
    
    

if __name__ == '__main__':
    main()
