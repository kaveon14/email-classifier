import tensorflow as tf
import numpy


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_example_object(data_record):
    string_list = data_record.split(",")
    label = string_list[len(string_list)-1]
    # int_values = tf.train.Int64List(value = [int_val])

    string_list = string_list[:len(string_list)-1]


    l = list(map(float, string_list))
    n = numpy.array(l)
    # n = n

    feature_key_value_pair = {
        'label' : _int64_feature(int(label)),
        'email' : _floats_feature(n)
    }

    features = tf.train.Features(feature = feature_key_value_pair)
    example = tf.train.Example(features = features)
    return example


def write_tfrecord(data_arr, filename):
    with tf.python_io.TFRecordWriter(filename) as tfwriter:
        # Iterate through all records
        for data_record in data_arr:
            example = get_example_object(data_record)
            s = example.SerializeToString()
            # Append each example into tfrecord
            tfwriter.write(s)


def get_data_arr():
    data_arr = []
    with open('spambase/spambase.data', 'r') as file:
        for line in file:
            data_arr.append(line)
    return data_arr


def create_full_dataset():
    data_arr = get_data_arr()
    MAX_ENTRIES = len(data_arr)

    TRAINING_INDEX = int(MAX_ENTRIES * 0.7)
    training_data = data_arr[:TRAINING_INDEX]

    VALIDATION_INDEX = MAX_ENTRIES - int(MAX_ENTRIES * .15)
    validation_data = data_arr[TRAINING_INDEX:VALIDATION_INDEX]

    testing_data = data_arr[VALIDATION_INDEX:]

    training_tfr, validation_tfr, testing_tfr = 'training.tfrecord',\
                                                'validation.tfrecord', 'testing.tfrecord'

    write_tfrecord(training_data, training_tfr)
    write_tfrecord(validation_data, validation_tfr)
    write_tfrecord(testing_data, training_tfr)



    return (TRAINING_INDEX, training_tfr, validation_tfr, testing_tfr)