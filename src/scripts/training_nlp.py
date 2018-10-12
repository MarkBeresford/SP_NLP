from nlp_utils import *
import tensorflow as tf
from SAndPData import SAndPData
import logging
import sys


def train_rnn(features_train, features_test, labels_train, labels_test, sequence_length_train, sequence_length_test):
    logger.info('Starting Training.')
    x = tf.placeholder(tf.float32, [n_steps, max_time_series_length, n_inputs])
    y = tf.placeholder(tf.float32, [n_steps, max_time_series_length, n_oututs])
    seq_length = tf.placeholder(tf.int32, [n_steps])

    cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_oututs)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=seq_length)
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimzer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        logger.info('Starting Epochs.')
        for epoch in range(n_epochs):
            num_training_iterations = int(len(features_train) / BATCH_SIZE)
            for train_iteration in range(num_training_iterations):
                batch_features_train = get_next_batch(features_train, train_iteration)
                batch_labels_train = get_next_batch(labels_train, train_iteration)
                batch_sequence_lengths_train = get_next_batch(sequence_length_train, train_iteration)
                sess.run(training_op, feed_dict={x: batch_features_train, y: batch_labels_train, seq_length: batch_sequence_lengths_train})
            if epoch % 10 == 0:
                num_test_iterations = int(len(features_test) / BATCH_SIZE)
                test_mses = []
                for test_iteration in range(num_test_iterations):
                    batch_features_test = get_next_batch(features_test, test_iteration)
                    batch_labels_test = get_next_batch(labels_test, test_iteration)
                    batch_sequence_lengths_test = get_next_batch(sequence_length_test, test_iteration)
                    mse = loss.eval(feed_dict={x: batch_features_test, y: batch_labels_test, seq_length: batch_sequence_lengths_test})
                    test_mses.append(mse)
                average_mse = sum(test_mses) / len(test_mses)
                logger.info("Epoch: {} of {}, MSE: {}".format(epoch, n_epochs, average_mse))
        logger.info('Training complete!')
        logger.info('Saving Model to {}'.format(CHECKPOINT_FOLDER_PATH))
        if not os.path.isdir(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        saver.save(sess, CHECKPOINT_FOLDER_PATH)


def get_normised_features(sandp_data_features):
    sandp_np = SAndPData.convert_pandas_df_to_numpy_array(sandp_data_features)
    sandp_np_normalised = SAndPData.normalise_features(sandp_np)
    return sandp_np_normalised


def get_company_names(sandp_data_company_names):
    return SAndPData.convert_pandas_df_to_numpy_array(sandp_data_company_names)


def convert_cn_string_to_numbers(company_names):
    batch_size_start = 0
    sequence_lengths = []
    not_reached_end_of_company_list = True
    while not_reached_end_of_company_list:
        for num_points in range(min_time_series_length, max_time_series_length + 1):
            if batch_size_start + num_points < len(company_names):
                batch = company_names[batch_size_start:batch_size_start + num_points]
                batch_including_next_point = company_names[batch_size_start:batch_size_start + num_points + 1]
                if len(set(batch)) == 1 and len(set(batch_including_next_point)) == 1:
                    sequence_lengths.append(len(batch))
            elif batch_size_start == len(company_names):
                not_reached_end_of_company_list = False
        batch_size_start += 1
    return sequence_lengths


def extract_features_and_labels_and_sequence_lengths():
    logger.info('Reading In Data.')
    sandp_data = SAndPData(os.path.join(SRC_FOLDER_PATH, 'sandp500/test.csv'), feature_columns, 'Name')
    logger.info('Cleaning Data.')
    cleaned_data = get_normised_features(sandp_data.feature_columns)
    company_names = get_company_names(sandp_data.company_names)
    logger.info('Extracting Sequence Lengths.')
    sequence_lengths = convert_cn_string_to_numbers(company_names)
    logger.info('Extracting Features and Labels.')
    features, labels = extract_batches_and_labels(cleaned_data, sequence_lengths)
    return split_test_train_sets(features, labels, sequence_lengths)


def start_training():
    features_train, features_test, labels_train, labels_test, sequence_length_train, sequence_length_test = extract_features_and_labels_and_sequence_lengths()
    train_rnn(features_train, features_test, labels_train, labels_test, sequence_length_train, sequence_length_test)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    start_training()