import os
import tensorflow as tf
import semantic_visualization


def make_image_mask(images_file, annotations_file, class_names, width=224, height=224):
    # Convert image and mask files to tensors
    image_data = tf.io.read_file(images_file)
    annotation_data = tf.io.read_file(annotations_file)
    image = tf.image.decode_jpeg(image_data)
    annotation = tf.image.decode_jpeg(annotation_data)

    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width))
    annotation = tf.image.resize(annotation, (height, width))
    image = tf.reshape(image, (height, width, 3))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1))
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:, :, 0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))

    annotation = tf.stack(stack_list, axis=2)

    image = image / 127.5
    image -= 1

    return image, annotation, class_names


def get_dataset_paths(image_dir, label_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    labels_paths = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir)]
    return image_paths, labels_paths


def get_data(image_paths, labels_paths, class_names, train):
    BATCH_SIZE = 64
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_paths))
    dataset = dataset.map(lambda x, y: make_image_mask(x, y, class_names))
    if train:
        dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    if train:
        dataset = dataset.prefetch(-1)
    return dataset


if __name__ == '__main__':
    timage_path = 'dataset1/images_prepped_train/'
    tlabel_path = 'dataset1/annotations_prepped_train/'
    class_names = ['sky', 'building', 'column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence',
                   'vehicle', 'pedestrian', 'byciclist', 'void']
    timage_paths, tlabels_paths = get_dataset_paths(timage_path, tlabel_path)
    training_dataset = get_data(timage_paths, tlabels_paths, class_names, True)

    trainV = semantic_visualization.Visualization(training_dataset, 9)
