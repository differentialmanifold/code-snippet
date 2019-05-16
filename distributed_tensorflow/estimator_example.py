"""An example of training Estimator model with multi-worker strategies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_path', '/tmp/datasets/mnist.npz', 'dataset path for mnist')
tf.app.flags.DEFINE_string('model_path', '/tmp/tfestimator_example/', 'model path for mnist')

BUFFER_SIZE = 10000
BATCH_SIZE = 64
num_classes = 10


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


def input_fn(mode, input_context=None):
    (x_train, y_train), (x_test, y_test) = load_data(FLAGS.dataset_path)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    datasets_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    datasets_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    mnist_dataset = (datasets_train if mode == tf.estimator.ModeKeys.TRAIN else
                     datasets_test)

    def scale(image, label):
        image = tf.reshape(image, (784,))
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                            input_context.input_pipeline_id)

    result_dataset = mnist_dataset.map(scale)

    result_dataset = result_dataset.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=BUFFER_SIZE, count=1))

    return result_dataset.batch(BATCH_SIZE)


LEARNING_RATE = 1e-3


def model_fn(features, labels, mode):
    # Define a Keras Model.
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    loss = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=False)

    loss = tf.reduce_mean(loss)

    predicted_classes = tf.argmax(logits, 1)
    predicted_labels = tf.argmax(labels, 1)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=predicted_labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()))


def main(_):
    model_dir = FLAGS.model_path
    print('Using %s to store checkpoints.' % model_dir)

    config = tf.estimator.RunConfig(
        experimental_distribute=tf.contrib.distribute.DistributeConfig(
            train_distribute=tf.contrib.distribute.CollectiveAllReduceStrategy(),
            remote_cluster={"worker": ["hostname1:5000", "hostname2:5000"]}))
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.model_path, config=config)

    # Train and evaluate the model. Evaluation will be skipped if there is not an
    # "evaluator" job in the cluster.
    tf.estimator.train_and_evaluate(
        classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
        eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
