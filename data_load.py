# Created on Wed Mar  6 10:19:45 2019
# @author: ZX
import tensorflow as tf
import glob


class image_load(object):
    def __init__(self, image_path, label_path, img_height, img_width, batchsize, channel, shuffle=True):
        self.image_path = image_path
        self.label_path = label_path
        self.batch_size = batchsize
        self.shuffle = shuffle
        self.channel = channel
        self.width = img_width
        self.height = img_height

    def process_data(self, filename, label):
        image = tf.read_file(filename)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=self.channel),
            lambda: tf.image.decode_png(image, channels=self.channel))
        image = tf.image.resize_images(image, size=[self.height, self.width], method=0)
        image = tf.cast(image, tf.float32) / 255.0

        label = tf.read_file(label)
        label = tf.cond(
            tf.image.is_jpeg(label),
            lambda: tf.image.decode_jpeg(label, channels=self.channel),
            lambda: tf.image.decode_png(label, channels=self.channel))
        label = tf.image.resize_images(label, size=[self.height, self.width], method=0)
        label = tf.cast(label, tf.float32) / 255.0

        return image, label

    def batch_generator(self):
        train_dataset = tf.data.Dataset().from_tensor_slices((self.image_path, self.label_path))
        train_dataset = train_dataset.map(self.process_data)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.repeat()
        if self.shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_dataset = train_dataset.repeat()
        return train_dataset


if __name__ == '__main__':
    batch_size = 4
    image_height = 200
    image_width = 300
    pathone = glob.glob(r'D:\Documents\Project_lowlight\HDR\data2\train\1\*')
    pathtwo = glob.glob(r'D:\Documents\Project_lowlight\HDR\data2\train\2\*')
    dataset = image_load(pathone, pathtwo, batch_size, 1, shuffle=True)
    data = dataset.batch_generator()
    print(dataset)

