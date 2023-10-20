import os
import tensorflow as tf


def make_image_mask(images_file, annotations_file, class_names, width=224, height=224):
      
    # Convert image and mask files to tensors 
    image_data = tf.io.read_file(images_file)
    annotation_data = tf.io.read_file(annotations_file)
    image = tf.image.decode_jpeg(image_data)
    annotaion = tf.image.decode_jpeg(annotation_data)


    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width))
    annotaion = tf.image.resize(annotaion, (height, width))
    image = tf.reshape(image, (height, width, 3))
    annotaion = tf.cast(annotaion, dtype=tf.int32)
    annotaion = tf.reshape(annotaion, (height, width, 1)) 
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask =tf.equal(annotaion[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))

    annotaion = tf.stack(stack_list, axis=2)

    image = image/127.5
    image -= 1

    return image, annotaion



def get_dataset_paths(image_dir, label_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    labels_path = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir)]
    return image_paths, labels_path


def get_data(image_paths, lables_paths, train):
    BATCH_SIZE = 64
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, lables_paths))    
    dataset = dataset.map(make_image_mask)
    if train:
        dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    if train:
        dataset = dataset.prefetch(-1)
    return dataset



    
    
