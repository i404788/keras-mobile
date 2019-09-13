import tensorflow as tf
import keras 
import numpy as np

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    tensor = (tensor[0] * 255).astype('uint8')
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImageComparison(keras.callbacks.Callback):
    """
    For AE-like models only
    """
    def __init__(self, log_dir, tag, x_test):
        super().__init__() 
        self.tag = tag
        self.test_set = x_test
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        sample = np.array([self.test_set[np.random.randint(0, self.test_set.shape[0])]])
        img = self.model.predict(sample)
        # img = data.astronaut()
        # Do something to the image
        # img = (255 * skimage.util.random_noise(img)).astype('uint8')

        org_image = make_image(sample)
        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        org_summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag + '-org', image=org_image)])
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, epoch)
        writer.add_summary(org_summary, epoch)
        writer.close()

        return