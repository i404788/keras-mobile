import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import keras 
import keras.backend as K
import numpy as np
import os
from PIL import Image

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    tensor = (tensor[0] * 255).astype('uint8')
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

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height x channels"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    img_c = images.shape[3]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots, img_w * n_plots, img_c))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w, :] = this_img
    
    return spriteimage

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



class TensorBoardModelEmbedding(keras.callbacks.Callback):
    """
    If no y_data available, make it the same shape as (x_data[0], ...model_output)
    """
    def __init__(self, log_dir, embeddings_freq, embedding_model, x_data, y_data, is_image=False, image_shape=[28,28], **kwargs):
        self._model = embedding_model
        self.freq = embeddings_freq
        self.log_dir = os.path.join(os.path.abspath(log_dir), 'projector')

        self.x_data = x_data
        self.is_image = is_image

        self.sess = K.get_session()
        self.tensor = [tf.Variable(tf.zeros((y_data.shape[0],) + tuple(self._model.get_layer(index=-1).output_shape[1:]), name='model_response'))]
        self.saver = tf.train.Saver(self.tensor)

        os.makedirs(self.log_dir, exist_ok=True)

        if is_image:
            sprites = create_sprite_image(x_data)
            if sprites.shape[0] > 8128:
                print("WARNING: Tensorboard Embedding spritesheet might be too big")
            if sprites.shape[-1] is 1:
                _mode = 'L'
                sprites = sprites.reshape(sprites.shape[0],sprites.shape[1])
            if sprites.shape[-1] is 3:
                _mode = 'RGB'
            if sprites.shape[-1] is 4:
                _mode = 'RGBA'
            print(sprites.shape)
            img = Image.fromarray((255 * sprites).astype('uint8'), mode=_mode)
            img.save(os.path.join(self.log_dir, 'spritesheet.png'), format='PNG')

        self.writer = tf.summary.FileWriter(self.log_dir)
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name= self.tensor[0].name
        if y_data is not None:
            meta_path = os.path.join(self.log_dir, 'metadata.tsv')
            with open(meta_path,'w') as f:
                f.write("Index\tLabel\n")
                for index,label in enumerate(y_data):
                    f.write("%d\t%d\n" % (index,label))
            embed.metadata_path = meta_path

        if self.is_image:
            embed.sprite.image_path = os.path.join(self.log_dir, 'spritesheet.png')
            embed.sprite.single_image_dim.extend(image_shape)

        projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.freq == 0:
            response = [self._model.predict(self.x_data)]
            K.batch_set_value(list(zip(self.tensor, response)))
            self.saver.save(self.sess, os.path.join(self.log_dir, 'keras_embedding.ckpt'), epoch)