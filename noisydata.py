import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,Input
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)
noise = 0.5
X_train_noisy = X_train + noise*np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
test_noisy = test + noise*np.random.normal(loc=0.0, scale=1.0, size=test.shape) 

X_train_noisy = X_train_noisy.astype('float32') / 255.
test_noisy = test_noisy.astype('float32') / 255.



def preprocess(x):
    x = x.astype('float32') / 255.
    return x.reshape(-1, np.prod(x.shape[1:])) # flatten
X_train_noisy = preprocess(X_train_noisy)
test_noisy  = preprocess(test_noisy)
print(X_train_noisy.shape)
print(test_noisy.shape)

autoencoder.fit(X_train_noisy, X_train_noisy,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(test_noisy, test_noisy))

encoded_imgs = encoder.predict(X_train_noisy)
decoded_imgs = decoder.predict(encoded_imgs)

d = decoded_imgs[0]
d.shape = (28,28)
plt.imshow(d,cmap='gray')


#g = sns.countplot(Y_train)


# Encode labels to one hot vector
Y_train = to_categorical(Y_train, num_classes = 10)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


