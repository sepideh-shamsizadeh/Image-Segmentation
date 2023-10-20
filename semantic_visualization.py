import math
import numpy as np
import seaborn as sns
import matplotlib as plt

class visualization:
    def __init__(self, dataset, num):
        self.dataset = dataset.unbatch().shuffle(buffer_size=100)
        self.num = num
        self.show_dataset()
    
    def show_img_ann(self, image, annotation):
        new_ann = np.argmax(annotation, axis=2)
        seg_img = np.zeros((new_ann.shape[0], new_ann.shape[1], 3)).astype('float')
        colors = sns.color_palette(None, len(12))
        for c in range(12):
            segc = (new_ann == c)
            seg_img[:,:,0] += segc*(colors[c][0] * 255.0)
    
    def show_dataset(self):
        plt.figure(figsize=(25, 15))
        plt.title("Image and Annotation")
        plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

        for idx, (image, annotation) in enumerate(self.dataset.take(self.num)):
            plt.subplot(int(math.sqrt(self.num)), int(math.sqrt(self.num)), idx+1)
            plt.yticks([])
            plt.xticks([])
            # .numpy() convert a tensor or array-like object into a NumPy array.
            self.show_img_ann(image.numpy(), annotation.numpy())
