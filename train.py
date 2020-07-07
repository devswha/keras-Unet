from preprocessing import *
from model import unet
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tifffile import imsave as tifsave


class myUnet(object):

    def __init__(self, img_rows=512, img_cols=512, save_path="./results/"):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.save_path = save_path

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def train(self, load_pretrained):
        print("loading data")
        model_name = 'my_model.h5'
        log_dir = "logs/000"
        logging = TensorBoard(log_dir=log_dir)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        if load_pretrained:
            model = load_model(model_name)
            model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            model_checkpoint = ModelCheckpoint('unet.h5', monitor='val_loss', verbose=1, save_best_only=True)
            model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=30, verbose=1,
                      validation_split=0.2, shuffle=True, callbacks=[logging, model_checkpoint, reduce_lr])
            model.save(model_name)
        else:
            model = unet()
            model.summary()
            model_checkpoint = ModelCheckpoint('unet.h5', monitor='val_loss', verbose=1, save_best_only=True)
            model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=30, verbose=1,
                      validation_split=0.2, shuffle=True,
                      callbacks=[logging, model_checkpoint, reduce_lr, early_stopping])
            model.save(model_name)

    def test(self):
        model_name = 'my_model.h5'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        model = load_model(model_name)
        imgs_mask_test = model.predict(imgs_test, batch_size=2, verbose=1)
        np.save(self.save_path + "imgs_mask_test.npy", imgs_mask_test)

        print("array to image")
        imgs = np.load(self.save_path + "imgs_mask_test.npy")
        total = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            total.append(img)
        np_total = np.array(total)
        tifsave("./prediction.tif", np_total)


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train(load_pretrained=False)
    myunet.test()
