# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse

from keras.datasets import cifar10
from keras.models import load_model
from keras.layers import Input
import scipy.misc

from configs import bcolors
from utils import *

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in CIFAR-10 dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# CIFAR-10 input image dimensions
img_rows, img_cols = 32, 32

(_, _), (x_test, _) = cifar10.load_data()
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple CIFAR-10 models
K.set_learning_phase(0)
model1 = load_model('Model1.h5')
model2 = load_model('Model2.h5')

# init coverage table
model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)

# ==============================================================================================
# start gen inputs
img_paths = image.list_pictures('./seeds/', ext='png')  # Assuming seed images are in png format for CIFAR-10
for _ in xrange(args.seeds):
    gen_img = preprocess_image(random.choice(img_paths))
    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2 = model1.predict(gen_img), model2.predict(gen_img)
    label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
    if label1 != label2:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}'.format(decode_label(pred1),
                                                                                        decode_label(
                                                                                            pred2)) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        scipy.misc.imsave('./generated_inputs/' + 'already_differ_' + decode_label(pred1) + '_' + decode_label(
            pred2) + '.png', gen_img_deprocessed)
        continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('output_layer').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('dense_2').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('output_layer').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('dense_2').output[..., orig_label])

    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    layer_output = (loss1 + loss2) + args.weight_nc * (loss1_neuron + loss2_neuron)
    print("--------------------")
    print('layer_output:', layer_output)
    # for adversarial image generation
    final_loss = K.mean(layer_output)
    # grads = normalize(K.gradients(final_loss, input_tensor)[0])
    try:
        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])
    except ValueError:
        continue

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss1_neuron, loss2_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):
        loss_value1, loss_value2, loss_neuron1, loss_neuron2, grads_value = iterate([gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        gen_img += grads_value * args.step
        pred1, pred2 = model1.predict(gen_img), model2.predict(gen_img)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])

        if label1 != label2:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            scipy.misc.imsave(
                './generated_inputs/' + args.transformation + '_' + decode_label(pred1) + '_' + decode_label(
                    pred2) + '.png', gen_img_deprocessed)
            scipy.misc.imsave(
                './generated_inputs/' + args.transformation + '_' + decode_label(pred1) + '_' + decode_label(
                    pred2) + '_orig.png', orig_img_deprocessed)
            break
