import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def print_model(filename, epoch, batch_size, dropout, double):
    # Define number of features, labels, and hidden
    num_features = 28 # 14 pairs of (x, y) keypoints
    num_labels = 5
    
    '''
    build_model

    # Builds an ANN model for keypoint predictions
    @params {list of labels} image prediction labels to be tested
    @params {integer} number of features
    @params {integer} number of labels as output
    @params {integer} number of hidden layers
    '''
    model = Sequential()
    model.add(Dense(60, input_shape=(num_features,)))
    model.add(Dense(30, activation='relu'))
    if double:
        model.add(Dense(30, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # Print model
    with open(f'{filename}.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file=f'{filename}.png', show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
    print_model('initial_pose_model_smallest_parameters', 100, 10, 0.1, False)
    print_model('initial_pose_model_largest_parameters', 250, 100, 0.3, True)