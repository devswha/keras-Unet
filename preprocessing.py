from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
from PIL import Image, ImageSequence
import os


class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="./data"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path

    def create_train_data(self):
        print('-' * 30)
        print('Creating train images...')
        print('-' * 30)

        # Load images and convert to npy
        i = 0
        j = 0
        imgs = os.listdir(self.data_path + "/raw/images/")
        aug_imgs = os.listdir(self.data_path + "/aug/images/")
        print("original images", len(imgs))
        print("augmented images", len(aug_imgs))
        img_datas = np.ndarray((len(imgs) + len(aug_imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        img_labels = np.ndarray((len(imgs) + len(aug_imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            img = load_img(self.data_path + "/raw/images/" + imgname, color_mode="grayscale")
            label = load_img(self.data_path + "/raw/labels/" + imgname, color_mode="grayscale")
            img_datas[j] = img_to_array(img)
            img_labels[j] = img_to_array(label)
            j += 1
            if j % len(imgs) == 0:
                print('Done: {0}/{1} images'.format(j, len(imgs)))
        for imgname in aug_imgs:
            img = load_img(self.data_path + "/aug/images/" + imgname, color_mode="grayscale")
            label = load_img(self.data_path + "/aug/labels/" + imgname, color_mode="grayscale")
            img_datas[i + len(imgs)] = img_to_array(img)
            img_labels[i + len(imgs)] = img_to_array(label)
            i += 1
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(aug_imgs)))

        print('loading done')
        np.save(self.data_path + '/npy/imgs_train.npy', img_datas)
        np.save(self.data_path + '/npy/imgs_mask_train.npy', img_labels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)

        # Create directory
        if not os.path.exists(self.data_path + "/test/images"):
            os.makedirs(self.data_path + "/test/images")
        if not os.path.exists(self.data_path + "/test/labels"):
            os.makedirs(self.data_path + "/test/labels")

        # Split isbi tif image&label to single frame of png images
        isbi_img = Image.open(self.data_path + "/test-volume.tif")  # raw image from isbi dataset
        for i, page in enumerate(ImageSequence.Iterator(isbi_img)):
            page.save(self.data_path+"/test/images/" + str(i) + ".png")

        imgs = os.listdir(self.data_path + "/test/images/")
        img_datas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        i = 0
        for imgname in imgs:
            img = load_img(self.data_path + "/test/images/" + imgname, color_mode="grayscale")
            img_datas[i] = img_to_array(img)
            i += 1
            if i % len(imgs) == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
        print('loading done')
        np.save(self.data_path + '/npy/imgs_test.npy', img_datas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.data_path + "/npy/imgs_train.npy")
        imgs_mask_train = np.load(self.data_path + "/npy/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255  # RGB 0~1
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.data_path + "/npy/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test


if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()
