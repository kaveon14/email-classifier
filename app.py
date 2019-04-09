import tensorflow as tf
import numpy
from tensorflow import keras
import tensor.part_data as data


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

def get_data_arr():
    data_arr = []
    with open('spambase/spambase.data', 'r') as file:
        for line in file:
            data_arr.append(line)
    return data_arr


def write_tfrecord(data_arr):
    with tf.python_io.TFRecordWriter('training-data.tfrecord') as tfwriter:
        # Iterate through all records
        for data_record in data_arr:
            example = get_example_object(data_record)
            s = example.SerializeToString()
            # Append each example into tfrecord
            tfwriter.write(s)
            


def create_record():
    write_tfrecord(get_data_arr())

def print_records(filename):
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        print(example)

        # Exit after 1 iteration as this is purely demonstrative.
        break


filename = 'training-data.tfrecord'

def _parse_function(proto):
    keys_to_features = {'label': tf.FixedLenFeature([], tf.int64),
                       'email': tf.FixedLenFeature([57], tf.float32)}

    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # parsed_features['email_data'] = tf.decode_raw(
    #     parsed_features['image'], tf.float64)

    return  parsed_features['email'], parsed_features['label']



SHUFFLE_BUFFER = 2*4601 # needs to be greater that or equal to the size of the dataset
BATCH_SIZE = 64
#50 with batch:32
# 50 with batch:64

def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    email, label = iterator.get_next()

    # 0 : not spam
    # 1 : spam
    # [0,1] spam

    # [1,0] not spam

    label = tf.one_hot(indices=label, depth=2)

    return email, label




# create_record()
training_records, training_tfr, validation_tfr, testing_tfr = data.create_full_dataset()

training_email, training_label = create_dataset(training_tfr)

validation_email, validation_label = create_dataset(validation_tfr)

testing_email, testing_label = create_dataset(testing_tfr)






















SUM_OF_ALL_DATASAMPLES = 4601
EPOCHS = 500

STEPS_PER_EPOCH = int((SUM_OF_ALL_DATASAMPLES) / BATCH_SIZE)

email, label = create_dataset(filename)


model_input = keras.layers.Input(tensor=email)

x = keras.layers.Dense(units=57,input_dim=57, activation='linear', use_bias=True)(model_input)# wtf is this doing
y = keras.layers.Dense(units=27, activation='linear', use_bias=True)(x) # added another layer
z = keras.layers.Dense(units=13, activation='linear', use_bias=True)(y) # added another layer, .33
f = keras.layers.Dense(units=7, activation='linear', use_bias=True)(z) # added another layer, .34d = keras.layers.Dropout(rate=.3,seed=2)(x), .9466 with drop
d = keras.layers.Dropout(rate=.1,seed=2)(f)#.93
model_output = keras.layers.Dense(units=2, activation=tf.math.softmax)(d) # .37

#Create your model
train_model = keras.models.Model(inputs=model_input, outputs=model_output)


#Compile your model
train_model.compile(optimizer='adam',
                    # loss=tf.keras.losses.categorical_crossentropy,
                    loss=tf.nn.softmax_cross_entropy_with_logits_v2,
                    # loss='binary_crossentropy',
                    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=.5)])

#Train the model
history = train_model.fit(x=email, y=label, epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH)
