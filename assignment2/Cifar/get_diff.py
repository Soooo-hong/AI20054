import numpy as np
import random
import imageio
import tensorflow as tf
from keras.datasets import cifar10
from Model_new_1_1 import Model1 
from Model_new_2_2 import Model2 
from Model_new_3_3 import Model3 
import imageio
import tensorflow as tf
from configs import bcolors
from utils import *
import cv2

def run_deepxplore_like_test(args, x_test, model1, model2, model3,
                              model_layer_dict1, model_layer_dict2, model_layer_dict3,
                              update_coverage, neuron_covered, neuron_to_cover,
                              deprocess_image, normalize, constraint_light,
                              constraint_occl, constraint_black, bcolors):
    idx = 1
    for _ in range(args['seeds']):
        gen_img_np = np.expand_dims(random.choice(x_test), axis=0).astype(np.float32)
        orig_img = gen_img_np.copy()
        gen_img = tf.Variable(gen_img_np)
        gen_img = tf.convert_to_tensor(gen_img)
        
        # Initial predictions
        pred1 = model1(gen_img)
        pred2 = model2(gen_img)
        pred3 = model3(gen_img)
        label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])
        
  
        if not (label1 == label2 == label3):
            print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2, label3) + bcolors.ENDC)
            update_coverage(gen_img, model1, model_layer_dict1, args['threshold'])
            update_coverage(gen_img, model2, model_layer_dict2, args['threshold'])
            update_coverage(gen_img, model3, model_layer_dict3, args['threshold'])
            # gen_img = tf.clip_by_value(gen_img, 0.0, 1.0)

            gen_img_deprocessed = deprocess_image(gen_img.numpy())
            filename = "./generated_inputs/already_differ_index{}_{}_{}_{}.png".format(idx,label1, label2, label3)
            # cv2.imwrite(filename, gen_img)
            cv2.imwrite(filename, gen_img_deprocessed)
            idx += 1 
            continue

        orig_label = label1
        for iters in range(args['grad_iterations']):
            with tf.GradientTape() as tape:
                layer_name1, index1 = neuron_to_cover(model_layer_dict1)
                layer_name2, index2 = neuron_to_cover(model_layer_dict2)
                layer_name3, index3 = neuron_to_cover(model_layer_dict3)

        
                tape.watch(gen_img)
                # pred1 = model1(gen_img)
                # pred2 = model2(gen_img)
                # pred3 = model3(gen_img)
                pred1 = get_before_softmax_output(model1, gen_img)
                pred2 = get_before_softmax_output(model2, gen_img)
                pred3 = get_before_softmax_output(model3, gen_img)
                if args['target_model'] == 0:
                    loss1 = -args['weight_diff'] * tf.reduce_mean(pred1[:, orig_label])
                    loss2 = tf.reduce_mean(pred2[:, orig_label])
                    loss3 = tf.reduce_mean(pred3[:, orig_label])
                elif args['target_model'] == 1:
                    loss1 = tf.reduce_mean(pred1[:, orig_label])
                    loss2 = -args['weight_diff'] * tf.reduce_mean(pred2[:, orig_label])
                    loss3 = tf.reduce_mean(pred3[:, orig_label])
                elif args.target_model == 2:
                    loss1 = tf.reduce_mean(pred1[:, orig_label])
                    loss2 = tf.reduce_mean(pred2[:, orig_label])
                    loss3 = -args['weight_diff'] * tf.reduce_mean(pred3[:, orig_label])

                intermediate_model1 = tf.keras.Model(inputs=model1.input, outputs=model1.get_layer(layer_name1).output)
                intermediate_model2 = tf.keras.Model(inputs=model2.input, outputs=model2.get_layer(layer_name2).output)
                intermediate_model3 = tf.keras.Model(inputs=model3.input, outputs=model3.get_layer(layer_name3).output)

                inter_out1 = intermediate_model1(gen_img)
                inter_out2 = intermediate_model2(gen_img)
                inter_out3 = intermediate_model3(gen_img)

                loss1_neuron = tf.reduce_mean(inter_out1[..., index1])
                loss2_neuron = tf.reduce_mean(inter_out2[..., index2])
                loss3_neuron = tf.reduce_mean(inter_out3[..., index3])

                layer_output = (loss1 + loss2 + loss3) + args['weight_nc'] * (loss1_neuron + loss2_neuron + loss3_neuron)
                final_loss = tf.reduce_mean(layer_output)

            grads = tape.gradient(final_loss, gen_img)
            grads = grads / (tf.reduce_max(tf.abs(grads)) + 1e-8)


            # Constraint 적용
            if args['transformation'] == 'light':
                grads_value = constraint_light(grads)
            elif args['transformation'] == 'occl':
                grads_value = constraint_occl(grads, args['start_point'], args['occlusion_size'])
            elif args['transformation'] == 'blackout':
                grads_value = constraint_black(grads)

            # gen_img.assign_add(grads * args['step'])
            step_size = args['step'] * (0.99 ** iters)
            gen_img = gen_img + grads_value * step_size
            gen_img = tf.clip_by_value(gen_img, 0, 1)
            # Prediction 업데이트 후 조건 확인
            pred1 = model1(gen_img)
            pred2 = model2(gen_img)
            pred3 = model3(gen_img)
            predictions1 = np.argmax(pred1[0])
            predictions2 = np.argmax(pred2[0])
            predictions3 = np.argmax(pred3[0])
            if not (predictions1 == predictions2 == predictions3):
                update_coverage(gen_img, model1, model_layer_dict1, args['threshold'])
                update_coverage(gen_img, model2, model_layer_dict2, args['threshold'])
                update_coverage(gen_img, model3, model_layer_dict3, args['threshold'])

                gen_img_deprocessed = deprocess_image(gen_img.numpy())
                orig_img_deprocessed = deprocess_image(orig_img)

                filename_gen = './generated_inputs/{}_{}_{}_{}.png'.format(args['transformation'], predictions1, predictions2, predictions3)
                filename_orig = './generated_inputs/{}_{}_{}_{}_orig.png'.format(args['transformation'], predictions1, predictions2, predictions3)
                print(bcolors.OKBLUE + 'Found new input that causes different outputs: {}, {}, {}'.format(predictions1, predictions2, predictions3) + bcolors.ENDC)
                imageio.imwrite(filename_gen, gen_img_deprocessed)
                imageio.imwrite(filename_orig, orig_img_deprocessed)
                print(f"Iteration {iters}: Predictions - Model1: {predictions1}, Model2: {predictions2}, Model3: {predictions3}")

                break

if __name__ == "__main__":
    # Example usage
    args = {
        'seeds': 20,
        'grad_iterations': 50,
        'weight_diff': 1,
        'target_model': 1,
        'transformation': 'occl',
        'step': 10,
        'threshold': 0,
        'start_point': (0, 0),
        'occlusion_size': (50, 50),
        'weight_nc': 0.1,
    }
    random.seed(42)
    img_rows, img_cols = 224, 224
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    (_, _), (x_test, _) = cifar10.load_data()
    x_test_resized = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_test])
    x_test = x_test_resized.reshape(x_test_resized.shape[0], img_rows, img_cols, 3)
    x_test = x_test.astype('float32') / 255.0
    input_shape = (img_rows, img_cols, 3)
    model1 = Model1(input_tensor=input_tensor)
    model2 = Model2(input_tensor=input_tensor)
    model3 = Model3(input_tensor=input_tensor)
    model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)
    
    run_deepxplore_like_test(args, x_test, model1, model2, model3,
                              model_layer_dict1, model_layer_dict2, model_layer_dict3,
                              update_coverage, neuron_covered, neuron_to_cover,
                              deprocess_image, normalize, constraint_light,
                              constraint_occl, constraint_black, bcolors)