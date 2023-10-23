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

if __name__ == '__main__':
    model = segmentation()
    print(model.summary)
    sgd = tf.keras.optimizers.SGD(learning_rate=1E-2, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    # number of training images
    train_count = 367

    # number of validation images
    validation_count = 101

    EPOCHS = 170
    BATCH_SIZE = 64

    steps_per_epoch = train_count//BATCH_SIZE
    validation_steps = validation_count//BATCH_SIZE

    class_names = ['sky', 'building', 'column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence',
                'vehicle', 'pedestrian', 'byciclist', 'void']
    
    timage_path = 'dataset1/images_prepped_train/'
    tlabel_path = 'dataset1/annotations_prepped_train/'

    timage_paths, tlabels_paths = pc.get_dataset_paths(timage_path, tlabel_path)
    training_dataset = pc.get_data(timage_paths, tlabels_paths, class_names, True)

    # trainV = pc.semantic_visualization.Visualization(training_dataset, 9)

    vimage_path = 'dataset1/images_prepped_test/'
    vlabel_path = 'dataset1/annotations_prepped_test/'
    vimage_paths, vlabels_paths = pc.get_dataset_paths(vimage_path, vlabel_path)
    validation_dataset = pc.get_data(vimage_paths, vlabels_paths, class_names, False)

    # validationV = pc.semantic_visualization.Visualization(validation_dataset, 9)

    print(training_dataset.take(1))
    print(validation_dataset.take(1))
    history = model.fit(training_dataset, 
                        steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps, epochs=EPOCHS)