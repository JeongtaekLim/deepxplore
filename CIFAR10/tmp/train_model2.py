# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras import optimizers, callbacks
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.datasets import cifar10
from CIFAR10.resnet50_32x32 import ResNet50


def Model2(input_tensor=None, train=False):
    WANNAFASTTRAINING = 0
    img_width, img_height = 32, 32
    batch_trainsize = 32
    batch_testsize = 32
    nb_epoch = 5
    learningrate = 1e-3
    momentum = 0.8
    num_classes = 10

    if train:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)

        if WANNAFASTTRAINING == 1:
            X_train = X_train[:1500]
            y_train = y_train[:1500]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        input_tensor = Input(shape=(img_width, img_height, 3))
    elif input_tensor is None:
        print('you have to provide input_tensor when testing')
        exit()

    previouslytrainedModelpath = './trained_models/resnet50model1.h5'

    if os.path.isfile(previouslytrainedModelpath):
        print('Loading previously trained model...')
        custom_resnet_model = load_model(previouslytrainedModelpath)
        print(previouslytrainedModelpath + ' successfully loaded!')
    else:
        print('Initializing resnet50 model')
        model = ResNet50(input_tensor=input_tensor, include_top=True, weights='imagenet')
        x = model.get_layer('res5a_branch2a').input
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.5)(x)
        output_tensor = Dense(num_classes, activation='softmax', name='output_layer')(x)
        custom_resnet_model = Model(inputs=input_tensor, outputs=output_tensor)

    for layer in custom_resnet_model.layers:
        layer.trainable = True

    custom_resnet_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=learningrate, momentum=momentum),
        metrics=['accuracy']
    )

    if train:
        tb = callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True)
        filepath = "./trained_models/model1_-{epoch:02d}-{val_accuracy:.2f}_"
        checkpoint = callbacks.ModelCheckpoint(
            filepath + datetime.now().strftime('%Y-%m-%d_%H.%M.%S') + '.h5',
            monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1
        )

        t = time.time()
        custom_resnet_model.fit(
            X_train, y_train,
            batch_size=batch_trainsize,
            epochs=nb_epoch,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[tb, checkpoint]
        )
        print('Training time: %s' % (time.time() - t))

        loss, accuracy = custom_resnet_model.evaluate(X_test, y_test, batch_size=batch_testsize, verbose=1)
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

        custom_resnet_model.save('Model2.h5')
        print('Model2 saved.')
    else:
        custom_resnet_model.load_weights(previouslytrainedModelpath)
        print('Model2 loaded')

    return custom_resnet_model


if __name__ == '__main__':
    Model2(train=True)
