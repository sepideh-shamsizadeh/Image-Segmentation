import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageFont, PIL.ImageDraw

class Visualization:
    def __init__(self, dataset, num):
        self.dataset = dataset.unbatch().shuffle(buffer_size=100)
        self.num = num
        self.show_dataset()
    
    def show_img_ann(self, image, annotation):
        print("Inside show_img_ann")
        print("Image Shape:", image.shape)
        print("Annotation Shape:", annotation.shape)
        new_ann = np.argmax(annotation, axis=2)
        seg_img = np.zeros((new_ann.shape[0], new_ann.shape[1], 3)).astype('float')
        colors = sns.color_palette("tab10", n_colors=12)

        for c in range(12):
            segc = (new_ann == c)
            seg_img[:,:,0] += segc * (colors[c][0] * 255.0)
            seg_img[:,:,1] += segc * (colors[c][1] * 255.0)
            seg_img[:,:,2] += segc * (colors[c][2] * 255.0)
        
        image += 1 
        image *= 127.5
        image = np.uint8(image)
        images = [image, seg_img]
        widths = (img.shape[1] for img in images)
        heights = (img.shape[0] for img in images)
        total_width = sum(widths)
        max_height = max(heights)

        new_im = PIL.Image.new('RGB', (total_width, max_height))

        x_offset = 0

        for im in images:
            pil_image = PIL.Image.fromarray(np.uint8(im))
            new_im.paste(pil_image, (x_offset, 0))
            x_offset += im.shape[1]

        plt.imshow(new_im)
    

    def show_dataset(self):
        plt.figure(figsize=(25, 15))
        plt.title("Image and Annotation")
        plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

        for idx, data_tuple in enumerate(self.dataset.take(self.num)):
            if isinstance(data_tuple, tuple) and len(data_tuple) >= 2:
                image, annotation = data_tuple[:2]  # Take the first two values from the tuple
                plt.subplot(int(math.sqrt(self.num)), int(math.sqrt(self.num)), idx + 1)
                plt.yticks([])
                plt.xticks([])
                # .numpy() convert a tensor or array-like object into a NumPy array.
                self.show_img_ann(image.numpy(), annotation.numpy())
        plt.show()


