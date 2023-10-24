import numpy as np
import tensorflow as tf
import preprocessing_semantic_dataset as pc


def VGG_16(image_input):
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='conv1-1')(image_input)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='conv1-2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max1')(x)
    p1 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='conv2-1')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='conv2-2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max2')(x)
    p2 = x

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='conv3-1')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='conv3-2')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='conv3-3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max3')(x)
    p3 = x

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv4-1')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv4-2')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv4-3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max4')(x)
    p4 = x

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv5-1')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv5-2')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv5-3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max5')(x)
    p5 = x

    vgg = tf.keras.Model(image_input, p5)

    vgg.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    n = 4096

    c6 = tf.keras.layers.Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6")(p5)
    c7 = tf.keras.layers.Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7")(c6)

    # return the outputs at each stage. you will only need two of these in this particular exercise 
    # but we included it all in case you want to experiment with other types of decoders.
    return (p1, p2, p3, p4, c7)


def fcn8_decoder(convs, n_classes):
    f1, f2, f3, f4, f5 = convs
  
    # upsample the output of the encoder then crop extra pixels that were introduced
    o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False )(f5)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o)

    # load the pool 4 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f4
    o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)

    # add the results of the upsampling and pool 4 prediction
    o = tf.keras.layers.Add()([o, o2])
    print(o.shape)
    # upsample the resulting tensor of the operation you just did
    o = (tf.keras.layers.Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False ))(o)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f3
    o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)

    # add the results of the upsampling and pool 3 prediction
    o = tf.keras.layers.Add()([o, o2])
    
    # upsample up to the size of the original image
    o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False )(o)

    # append a softmax to get the class probabilities
    o = (tf.keras.layers.Activation('softmax'))(o)

    return o 


def segmentation():

    inputs = tf.keras.layers.Input(shape=(224,224,3,))
    convs = VGG_16(image_input=inputs)
    outputs = fcn8_decoder(convs, 12)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model



def train_model(training_dataset, validation_dataset):

    model = segmentation()
    print(model.summary)
    sgd = tf.keras.optimizers.SGD(learning_rate=1E-2, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    # number of training images
    train_count = 367



    EPOCHS = 170
    BATCH_SIZE = 64

    steps_per_epoch = train_count//BATCH_SIZE

    print(training_dataset.take(1))
    print(validation_dataset.take(1))
    history = model.fit(training_dataset, 
                        steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps, epochs=EPOCHS)
    
    print(history)
    return model

def evaluate_model(validation_dataset):
    y_true_segments = []
    y_true_images = []
    test_count = 64

    ds = validation_dataset.unbatch()
    ds = ds.batch(101)

    for image, annotation in ds.take(1):
        y_true_images = image
        y_true_segments = annotation


    y_true_segments = y_true_segments[:test_count, : ,: , :]
    y_true_segments = np.argmax(y_true_segments, axis=3)  

    return y_true_images, y_true_segments


def IOU_diceScore(y_true, y_pred):
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(12):
        intersection = np.sum((y_pred==i) * (y_true==i))
        y_true_area = np.sum((y_true==1))
        y_pred_area = np.sum((y_pred==i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (combined_area-intersection+smoothening_factor)
        class_wise_iou.append(iou)

        dice_score = 2 * ((intersection+smoothening_factor)) / (combined_area + smoothening_factor)
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


if __name__ == '__main__':

    timage_path = 'dataset1/images_prepped_train/'
    tlabel_path = 'dataset1/annotations_prepped_train/'
    class_names = ['sky', 'building','column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'byciclist', 'void']
    
    BATCH_SIZE = 64
    
    dp = pc.DataPreprocessing(timage_path, tlabel_path, class_names, True, bacth_size=BATCH_SIZE)
    training_dataset = dp.get_data()

    trainV = pc.semantic_visualization.Visualization(training_dataset, 9, class_names)

    vimage_path = 'dataset1/images_prepped_test/'
    vlabel_path = 'dataset1/annotations_prepped_test/'
    dp = pc.DataPreprocessing(vimage_path, vlabel_path, class_names, False, bacth_size=BATCH_SIZE)
    validation_dataset = dp.get_data()
    validationV = pc.semantic_visualization.Visualization(validation_dataset, 9, class_names)

    model = train_model(training_dataset, validation_dataset)
    y_true_images, y_true_segments = evaluate_model(validation_dataset)
    
    # number of validation images
    validation_count = 101
    validation_steps = validation_count//BATCH_SIZE

    results = model.predict(validation_dataset, steps=validation_steps)

    # for each pixel, get the slice number which has the highest probability
    results = np.argmax(results, axis=3)
    # input a number from 0 to 63 to pick an image from the test set
    integer_slider = 0

    # compute metrics
    iou, dice_score = IOU_diceScore(y_true_segments[integer_slider], results[integer_slider])  

    # visualize the output and metrics
    validationV.semantic_visualization.show_predictions(y_true_images[integer_slider], [results[integer_slider], y_true_segments[integer_slider]], ["Image", "Predicted Mask", "True Mask"], iou, dice_score)
    cls_wise_iou, cls_wise_dice_score = IOU_diceScore(y_true_segments, results)
    # print IOU for each class
    for idx, iou in enumerate(cls_wise_iou):
        spaces = ' ' * (13-len(class_names[idx]) + 2)
        print("{}{}{} ".format(class_names[idx], spaces, iou)) 

    # print the dice score for each class
    for idx, dice_score in enumerate(cls_wise_dice_score):
        spaces = ' ' * (13-len(class_names[idx]) + 2)
        print("{}{}{} ".format(class_names[idx], spaces, dice_score))