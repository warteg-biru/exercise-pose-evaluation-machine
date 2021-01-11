import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.schedules import PolynomialDecay

def print_model(filename, double):
    # Define number of features, labels, and hidden
    num_features = 28 # 14 pairs of (x, y) keypoints
    num_output = 1

    model = Sequential()
    model.add(Dense(60, input_shape=(num_features,)))
    model.add(Dense(30, activation='relu'))
    if double:
        model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_output, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # Print model
    with open(f'{filename}.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file=f'{filename}.png', show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
    print_model('binary_pose_model_smallest_parameters', False)
    print_model('binary_pose_model_largest_parameters', True)