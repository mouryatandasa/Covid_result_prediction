from keras.models import Sequential
from keras.layers import Dense

def build_ann(input_dim):
    model = Sequential()

    model.add(Dense(8, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
