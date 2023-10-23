import tensorflow as tf

def VGG_16(image_input):
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='conv1-1')(image_input)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='conv1-2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='max1')(x)
    p1 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='conv2-1')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='conv2-2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='max1')(x)
    p2 = x

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='conv3-1')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='conv3-2')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='conv3-3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='max1')(x)
    p3 = x

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv4-1')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv4-2')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv4-3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='max1')(x)
    p4 = x

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv5-1')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv5-2')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='conv5-3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='max1')(x)
    p5 = x

    vgg = tf.keras.Model(image_input, p5)

    vgg.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    n = 4096

    c6 = tf.keras.layers.Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6")(p5)
    c7 = tf.keras.layers.Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7")(c6)

    # return the outputs at each stage. you will only need two of these in this particular exercise 
    # but we included it all in case you want to experiment with other types of decoders.
    return (p1, p2, p3, p4, c7)


