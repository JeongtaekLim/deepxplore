# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def Model1(input_tensor=None, train=False):
    nb_classes = 10
    img_rows, img_cols, img_channels = 32, 32, 3

    if train:
        batch_size = 32
        nb_epoch = 10

        # CIFAR-10 데이터셋 불러오기
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # 데이터 전처리
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=(img_rows, img_cols, img_channels))
    elif input_tensor is None:
        print('you have to provide input_tensor when testing')
        exit()

    # CNN 모델 구성
    x = Conv2D(32, (3, 3), activation='relu', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout')(x)
    output_tensor = Dense(nb_classes, activation='softmax', name='output_layer')(x)

    model = Model(input_tensor, output_tensor)

    if train:
        # 모델 컴파일
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 체크포인트 설정
        checkpoint = ModelCheckpoint('tmp/Model1_checkpoint.h5', save_best_only=True, monitor='val_loss', mode='min')

        # 모델 학습
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epoch, batch_size=batch_size,
                  callbacks=[checkpoint])

        # 모델 저장
        model.save('Model1.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\nOverall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('Model1.h5')
        print('Model1 loaded')

    return model


if __name__ == '__main__':
    Model1(train=True)
