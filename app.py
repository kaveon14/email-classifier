import tensorflow as tf
import numpy
from tensorflow import keras


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
    with tf.python_io.TFRecordWriter('test-data.tfrecord') as tfwriter:
        # Iterate through all records
        for data_record in data_arr:
            example = get_example_object(data_record)
            s = example.SerializeToString()
            # Append each example into tfrecord
            tfwriter.write(s)


def create_record():
    write_tfrecord(get_data_arr())


def print_records(count, filename):
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        print(example)

        # Exit after 1 iteration as this is purely demonstrative.
        break

create_record()

filename = 'test-data.tfrecord'
# print_records(9, filename)



def _parse_function(proto):
    keys_to_features = {'label': tf.FixedLenFeature([1], tf.int64),
                       'email': tf.FixedLenFeature([57], tf.float32)}

    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # parsed_features['email_data'] = tf.decode_raw(
    #     parsed_features['image'], tf.float64)

    return  parsed_features['email'], parsed_features['label']



SHUFFLE_BUFFER = 4601 # needs to be greater that or equal to the size of the dataset
BATCH_SIZE = 60


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


    # skip for now
    # Bring your picture back in shape
    # email = tf.reshape(email,0)
    # Create a one hot array for your labels
    label = tf.one_hot(indices=label,depth=1)

    return email, label


create_dataset(filename)







SUM_OF_ALL_DATASAMPLES = 780
EPOCHS = 5

STEPS_PER_EPOCH = int((SUM_OF_ALL_DATASAMPLES * EPOCHS) / BATCH_SIZE)
#Get your datatensors
email, label = create_dataset(filename)






#Combine it with keras
model_input = keras.layers.Input(tensor=email)

#Build your network
x = keras.layers.Dense(units=57, activation=tf.math.sigmoid)(model_input)# wtf is this doing
y =  keras.layers.Dense(units=27, activation=tf.math.sigmoid)(x) # added another layer
model_output = keras.layers.Dense(units=1, activation='relu')(y)

#Create your model
 train_model = keras.models.Model(inputs=model_input, outputs=model_output)




#Compile your model
train_model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=[tf.keras.metrics.Accuracy()],
                    target_tensors=[label])

print(train_model)

#Train the model
train_model.fit(x=email, y=label, epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH)

#More Kerasstuff here
print('done  ')




# d = raw_dataset.take(10)
# print(d)
'''
iterator = raw_dataset.make_initializable_iterator()

next_element = iterator.get_next()
init_op = iterator.initializer








model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit()



sess = tf.Session()
for i in range(1):
    sess.run(init_op)

    # tensors??
    label = sess.run(next_element)
    print(label)

test
'''

