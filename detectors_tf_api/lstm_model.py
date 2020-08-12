import os
import time
import random

# Set minimum logging to level 3 to prevent noisy command line
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import collections
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from db_entity import get_dataset
from keypoints_extractor import pop_all

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set minimal logging level to Errors only
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define number of training steps
n_steps = 24

# Define Tensorflow checkpoint path
checkpoint_path = "/home/kevin/projects/exercise_pose_evaluation_machine/models/lstm_model/lstm_model.ckpt"

def LSTM_RNN(_X, _weights, _biases):
    # Model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])  
    
    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    
    
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    

    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    lstm_last_output = outputs[-1]
    
    

    print(tf.math.add(
        tf.matmul(lstm_last_output, _weights['out']),
        _biases['out'],
        name='predict'
    ))
    import time 
    time.sleep(20)
    return tf.math.add(
        tf.matmul(lstm_last_output, _weights['out']),
        _biases['out'],
        name='predict'
    )

def extract_batch_size(_train, _labels, _unsampled, batch_size):
    # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data. 
    # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
    # unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size    
    _unsampled = list(_unsampled)

    # Extract train labels and batch shape
    _train = np.array(_train)
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = []
    batch_labels = np.array([])

    for i in range(batch_size):
        # Loop index
        # index = random sample from _unsampled (indices)
        index = random.choice(_unsampled)
        _train_shape = np.array(_train[index]).shape
        batch_s.append(_train[index])
        batch_labels = np.append(batch_labels, _labels[index])
        _unsampled.remove(index)

    batch_s = np.array(batch_s)
    return batch_s, batch_labels, _unsampled

def one_hot(y_):
    # One hot encoding of the network outputs
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 2
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


# Define class type
CLASS_TYPE = [
    "push-up2",
]

# Train for each dataset
for type_name in CLASS_TYPE:
    # Get original dataset
    x, y = get_dataset(type_name)

    # Fill original class type with the label 1
    y = [1 for label in y]

    # Get negative dataset
    neg_x, neg_y = get_dataset("not-" + type_name)

    # Fill original class type with the label 1
    neg_y = [0 for label in neg_y]
    x.extend(neg_x)
    y.extend(neg_y)

    # Flatten X coodinates and filter
    x = np.array(x)
    _x = []
    _y = []
    for idx, data in enumerate(x):
        if len(data) == 24:
            data = [np.reshape(np.array(frames), (28)).tolist() for frames in data]
            _x.append(data)
            _y.append(y[idx])
    x = _x
    y = _y

    # Split to training and test dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    n_input = len(x_train[0][0])
    n_hidden = 22
    n_classes = 3
    decaying_learning_rate = True
    learning_rate = 0.0025 
    init_learning_rate = 0.005
    decay_rate = 0.96
    decay_steps = 100000

    global_step = tf.Variable(0, trainable=False)
    lambda_loss_amount = 0.0015

    epochs = 500
    training_data_count = len(x_train)

    training_iters = training_data_count * 50 # Loop 50 times on the dataset, ie 50 epochs
    batch_size = 178
    # Originally 512
    display_iter = batch_size * 8  # To show test set accuracy during training

    step = 1
    unsampled_indices = range(0, len(x_train))

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name="tf_data")
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
    if decaying_learning_rate:
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)

    #decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step) # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    time_start = time.time()
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    while step * batch_size <= training_iters:
        saver = tf.train.Saver()
        #print (sess.run(learning_rate)) #decaying learning rate
        #print (sess.run(global_step)) # global number of iterations
        if len(unsampled_indices) < batch_size:
            unsampled_indices = range(0,len(x_train)) 
        batch_xs, raw_labels, unsampled_indicies = extract_batch_size(x_train, y_train, unsampled_indices, batch_size)
        batch_ys = one_hot(raw_labels)
        
        # check that encoded output is same length as num_classes, if not, pad it 
        if len(batch_ys[0]) < n_classes:
            temp_ys = np.zeros((batch_size, n_classes))
            temp_ys[:batch_ys.shape[0],:batch_ys.shape[1]] = batch_ys
            batch_ys = temp_ys

        # Fit training using batch data
        _, loss, acc = sess.run([optimizer, cost, accuracy] , feed_dict={x: batch_xs, y: batch_ys})
        train_losses.append(loss)
        train_accuracies.append(acc)
        
        # Evaluate network only at some steps for faster training: 
        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            
            # To not spam console, show training accuracy/loss in this "if"
            print("Iter #" + str(step * batch_size) + \
                ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) + \
                ":   Batch Loss = " + "{:.6f}".format(loss) + \
                ", Accuracy = {}".format(acc))
            save_path = saver.save(sess, checkpoint_path)
            
            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run(
                [cost, accuracy], 
                feed_dict={
                    x: x_test,
                    y: one_hot(y_test)
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print("PERFORMANCE ON TEST SET:             " + \
                "Batch Loss = {}".format(loss) + \
                ", Accuracy = {}".format(acc))

        step += 1
    

    # Accuracy for test data
    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: x_test,
            y: one_hot(y_test)
        }
    )
    test_losses.append(final_loss)
    test_accuracies.append(accuracy)
    print("FINAL RESULT: " + \
        "Batch Loss = {}".format(final_loss) + \
        ", Accuracy = {}".format(accuracy))
    time_stop = time.time()
    print("TOTAL TIME:  {}".format(time_stop - time_start))

    predictions = sess.run(pred, feed_dict={
        x: np.array([x_test[2]]),
        y: one_hot(np.array([y_test[2]]))
    })
    print(np.argmax(predictions), y_test[2])
    # JAPHNE DID THIS, ASK HIM THE MADAFAKA