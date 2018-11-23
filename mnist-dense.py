# распознавание рукописных цифр из датасета mnist с помощью полносвязной НС

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128    # размер пакета
num_classes = 10    # кол-во классов
epochs = 5         # кол-во эпох обучения

# размеры входных изображений (в пикселах)
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train[0], y_train[:5])
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# преобразование формы исходных данных в одномерный вид
x_train = x_train.reshape((x_train.shape[0], img_rows * img_cols))
x_test = x_test.reshape((x_test.shape[0], img_rows * img_cols))
#print("shapes after convertion", x_train.shape, y_train.shape, 
        #x_test.shape, y_test.shape)

# нормирование значений цвета пикселов для перевода в диапазон [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#print(x_train[0])

# # convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("y_train after convert class vectors to binary class matrices: ", y_train[:10])

# создание модели НС
network = Sequential()
# добавление слоёв полносвязного типа
network.add(Dense(img_rows*img_cols, # кол-во нейронов
            activation='relu', # тип ф-ции активации
            input_shape=(img_rows*img_cols,))) # кол-во входов
# добавление 2-го слоя
network.add(Dense(num_classes, # кол-во нейронов
                activation='softmax')) # тип ф-ции активации
# кол-во входов опред-ся автоматически в зависимости от кол-ва выходов предыдущего слоя

# задание параметров и компиляция модели
network.compile(optimizer='rmsprop', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])
# обучение модели 
network.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

# оценка модели
test_loss, test_acc = network.evaluate(x_test, y_test, verbose=1)
print("test_loss, test_acc: ", test_loss, test_acc)
