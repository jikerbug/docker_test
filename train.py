
import tensorflow.keras as keras
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
DATA_PATH = "data.json"
SAVED_MODEL_PATH = 'model.h5'

LEARNING_RATE = 0.0001
EPOCHS = 40 # 전체 데이터를 얼마나 많이 순회할 것인지
BATCH_SIZE = 32 # 한 epochs 내에서 단위 역전파당 얼마나 많은 sample을 참고할 것인지
NUM_KEYWORDS = 10

def load_dataset(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)

    # extract inputs and targets

    X_array = np.array(data["MFCCs"])
    y_array = np.array(data["labels"])

    return X_array,y_array

def get_data_splits(data_path, test_size=0.1, test_validation=0.1):

    # load dataset
    X, y = load_dataset(data_path)

    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)


    # convert inputs from 2d to 3d arrays
    # (# segment, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]


    return  X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate, error='sparse_categorical_crossentropy'):

    # build network
    model = keras.Sequential() #nn that has sequential layers !! ㅎㅎ

    # conv layer 1
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu',
                                  input_shape=input_shape, # only needed for initial layer
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding="same"))


    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flatten the output and feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3)) # shoot down 30% of layers during training
    # drop out의 효과 : 모델이 적은 뉴런에만 의존하지 않고 여러 뉴런이 비슷한 영향력을 갖게된다

    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation='softmax')) # [0.1, 0.7, 0.2 ...] which prediction has more probability

    # compile the model
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=['accuracy']) # we`re gonna track accuracy during training

    # print model overview
    model.summary()

    return model



def main():

    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    print(f'x_train shape : {X_train.shape}')
    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # ( # segments, # coefficients 13, 1)because of CNN, 1 means grey scale image ( 3 is RGB )
    model = build_model(input_shape, LEARNING_RATE)


    # train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()