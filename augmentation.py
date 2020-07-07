import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img,array_to_img
from utills import *
from random import randint
from PIL import Image, ImageSequence
import os


class dataAugment(object):
    def __init__(self, out_rows, out_cols, data_path="./data"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path

    def augmentation(self):
        # read images
        print('-' * 30)
        print('Augment train images...')
        print('-' * 30)

        # Create directory
        if not os.path.exists(self.data_path + "/raw/images"):
            os.makedirs(self.data_path + "/raw/images")
        if not os.path.exists(self.data_path + "/raw/labels"):
            os.makedirs(self.data_path + "/raw/labels")
        if not os.path.exists(self.data_path + "/aug/images"):
            os.makedirs(self.data_path + "/aug/images")
        if not os.path.exists(self.data_path + "/aug/labels"):
            os.makedirs(self.data_path + "/aug/labels")
        if not os.path.exists(self.data_path + "/npy"):
            os.makedirs(self.data_path + "/npy")

        # Split isbi tif image&label to single frame of png images
        isbi_img = Image.open(self.data_path + "/train-volume.tif")  # raw image from isbi dataset
        for i, page in enumerate(ImageSequence.Iterator(isbi_img)):
            page.save(self.data_path+"/raw/images/" + str(i) + ".png")

        isbi_lbl = Image.open(self.data_path + "/train-labels.tif")  # raw image from isbi dataset
        for i, page in enumerate(ImageSequence.Iterator(isbi_lbl)):
            page.save(self.data_path+"/raw/labels/" + str(i) + ".png")

        train_imgs = glob.glob(self.data_path + "/raw/images/*.png")
        label_imgs = glob.glob(self.data_path + "/raw/labels/*.png")
        slices = len(train_imgs)
        if len(train_imgs) != len(label_imgs) or len(train_imgs) == 0:
            print("trains can't match labels")
            return 0

        print('Using real-time data augmentation. len: ', slices)
        # one by one augmentation
        batch_size = 30  # one frame for 30 images augment
        for i in range(slices):
            for b in range(batch_size):
                img_as_img = Image.open(self.data_path + "/raw/images/" + str(i) + ".png")
                lbl_as_img = Image.open(self.data_path + "/raw/labels/" + str(i) + ".png")
                img_as_np = np.asarray(img_as_img)
                lbl_as_np = np.asarray(lbl_as_img)

                # flip {0: vertical, 1: horizontal, 2: both, 3: none}
                flip_num = randint(0, 3)
                img_as_np = flip(img_as_np, flip_num)
                lbl_as_np = flip(lbl_as_np, flip_num)

                # Noise Determine {0: Gaussian_noise, 1: uniform_noise
                if randint(0, 1):
                    # Gaussian_noise
                    gaus_sd, gaus_mean = randint(0, 20), 0
                    img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
                    lbl_as_np = add_gaussian_noise(lbl_as_np, gaus_mean, gaus_sd)
                else:
                    # uniform_noise
                    l_bound, u_bound = randint(-20, 0), randint(0, 20)
                    img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)
                    lbl_as_np = add_uniform_noise(lbl_as_np, l_bound, u_bound)

                # Brightness
                pix_add = randint(-20, 20)
                img_as_np = change_brightness(img_as_np, pix_add)
                lbl_as_np = change_brightness(lbl_as_np, pix_add)

                # Elastic distort {0: distort, 1:no distort}
                sigma = randint(6, 12)
                # sigma = 4, alpha = 34
                img_as_np, seed = add_elastic_transform(img_as_np, alpha=34, sigma=sigma, pad_size=20)
                lbl_as_np, seed = add_elastic_transform(lbl_as_np, alpha=34, sigma=sigma, pad_size=20)

                # Crop the image
                in_size = 512
                out_size = 388
                img_height, img_width = img_as_np.shape[0], img_as_np.shape[1]
                pad_size = int((in_size - out_size)/2)
                img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
                lbl_as_np = np.pad(lbl_as_np, pad_size, mode="symmetric")
                y_loc, x_loc = randint(0, img_height-out_size), randint(0, img_width-out_size)
                img_as_np = cropping(img_as_np, crop_size=in_size, dim1=y_loc, dim2=x_loc)
                lbl_as_np = cropping(lbl_as_np, crop_size=in_size, dim1=y_loc, dim2=x_loc)

                # Normalize the image
                img_as_np = normalization2(img_as_np, max=1, min=0)
                img = img_as_np.reshape(img_as_np.shape[0], img_as_np.shape[1], 1)
                img = array_to_img(img)
                img.save(self.data_path + "/aug/images/" + str(30*i+b) + ".png")

                lbl = lbl_as_np.reshape(lbl_as_np.shape[0], lbl_as_np.shape[1], 1)
                lbl = array_to_img(lbl)
                lbl.save(self.data_path + "/aug/labels/" + str(30*i+b) + ".png")

            print(str(i+1))


if __name__ == "__main__":
    try:
        mydata = dataAugment(512, 512)
        mydata.augmentation()
    except RuntimeError as e:
        print(e)
