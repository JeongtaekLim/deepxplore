# -*- coding: utf-8 -*-
from keras.models import load_model

model = load_model('Model1.h5')

for layer in model.layers:
    try:
        print(layer.name, layer.activation)

    # for some layers there will not be any activation fucntion.
    except:
        pass

# To get the name of layers in the model.
layer_names = [layer.name for layer in model.layers]

# for model's summary and details.
model.summary()
