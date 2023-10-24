import os
import tensorflow as tf
import semantic_visualization

class DataPreprocessing:
    def __init__(self, image_dir, label_dir,class_names, train=True, bacth_size=64, width=224, height=224,):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.width = width
        self.height = height
        self.train = train
        self.batch = bacth_size
        self.image_paths, self.labels_paths = self.get_dataset_paths()


    def make_image_mask(self, images_file, annotations_file):
        # Convert image and mask files to tensors
        image_data = tf.io.read_file(images_file)
        annotation_data = tf.io.read_file(annotations_file)
        image = tf.image.decode_jpeg(image_data)
        annotation = tf.image.decode_jpeg(annotation_data)
        
        # Resize image and segmentation mask
        image = tf.image.resize(image, (self.height, self.width, ))
        annotation = tf.image.resize(annotation, (self.height, self.width, ))
        image = tf.reshape(image, (self.height, self.width, 3, ))
        annotation = tf.cast(annotation, dtype=tf.int32)
        annotation = tf.reshape(annotation, (self.height, self.width, 1, ))
        stack_list = []

        # Reshape segmentation masks
        for c in range(len(self.class_names)):
            mask = tf.equal(annotation[:, :, 0], tf.constant(c))
            stack_list.append(tf.cast(mask, dtype=tf.int32))

        annotation = tf.stack(stack_list, axis=2)

        image = image / 127.5
        image -= 1

        return image, annotation


    def get_dataset_paths(self):
        image_paths = [os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir)]
        labels_paths = [os.path.join(self.label_dir, fname) for fname in os.listdir(self.label_dir)]
        return image_paths, labels_paths


    def get_data(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels_paths))
        dataset = dataset.map(self.make_image_mask)
        if self.train:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch)
        dataset = dataset.repeat()
        if self.train:
            dataset = dataset.prefetch(-1)
        return dataset


if __name__ == '__main__':
    timage_path = 'dataset1/images_prepped_train/'
    tlabel_path = 'dataset1/annotations_prepped_train/'
    class_names = ['sky', 'building','column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'byciclist', 'void']
    dp = DataPreprocessing(timage_path, tlabel_path, class_names, True)
    training_dataset = dp.get_data()
    trainV = semantic_visualization.Visualization(training_dataset, 9)
