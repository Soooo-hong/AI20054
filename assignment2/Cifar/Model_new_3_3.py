import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta

def preprocess(image, label):
    image = tf.numpy_function(lambda img: cv2.resize(img, (224, 224)), [image], tf.uint8)
    image.set_shape((224, 224, 3))  # 크기 고정
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label[0], depth=10)  # (1,) → 스칼라로 바꿔서 one-hot 인코딩
    return image, label

def Model3(input_tensor=None, train=False):
    num_classes = 10
    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)

    if train:
        batch_size = 128
        epochs = 10

        # Load and preprocess CIFAR-10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Resize images to 224x224 for ResNet50
        x_train_resized = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_train])
        x_test_resized = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_test])

        x_train = x_train_resized.astype('float32') / 255.0
        x_test = x_test_resized.astype('float32') / 255.0

        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        input_tensor = Input(shape=input_shape)

    elif input_tensor is None:
        raise ValueError("input_tensor must be provided in inference mode")

    # Load ResNet50 base
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu', name='fc1')(x)
    x = Dense(num_classes, name='before_softmax')(x)
    output_tensor = Activation('softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=output_tensor)

    if train:
        model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

        # model.fit(
        #     x_train, y_train,
        #     validation_data=(x_test, y_test),
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     verbose=1
        # )
        model.fit(
            train_ds,
            validation_data=test_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

        model.save_weights('./ResNet50_CIFAR10_Model3.h5')

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    else:
        model.load_weights('./ResNet50_CIFAR10_Model3.h5')
        print('Model3 weights loaded successfully.')

    return model


if __name__ == '__main__':
    Model3(train=True)
