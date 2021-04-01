__author__ = 'Bastien Rigaud'

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np
import io
import cv2

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


class Add_Heatmap(Callback):
    def __init__(self, log_dir, validation_steps, validation_data=None, class_names=[], frequency=5, nb_images=5,
                 layerName=None, image_rows=512, image_cols=512):
        super(Add_Heatmap, self).__init__()
        if validation_data is None:
            AssertionError('Need to provide validation data')
        self.validation_data = iter(validation_data)
        self.validation_steps = validation_steps
        self.class_names = class_names
        self.nb_images = nb_images
        self.frequency = frequency
        self.layerName = layerName
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, 'val_heatmap'))

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4 and ('activation' in layer.name.lower() or 'pooling' not in layer.name.lower()):
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, classIdx, eps=1e-8):

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            # inputs = tf.cast(image, tf.float32)
            # (convOutputs, predictions) = self.gradModel(inputs)
            (convOutputs, predictions) = self.gradModel(image[None, ...])
            loss = predictions[0][:, classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[1], image.shape[0])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        # Closing the figure prevents it from being displayed
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, (self.image_rows, self.image_cols), method='bilinear',
                                preserve_aspect_ratio=True)
        image = tf.image.resize_with_crop_or_pad(image, target_height=self.image_rows, target_width=self.image_cols)
        image = tf.cast(image, np.uint8)
        return image

    def plot_heatmap(self, heatmap, image, gt_id, pred_id, prob, alpha=0.5, eps=1e-8, add_colormap=False):
        figure = plt.figure(figsize=(8, 8))
        image = image.numpy()
        numer = image - np.min(image)
        denom = (image.max() - image.min()) + eps
        image = numer / denom
        image = (image * 255).astype("uint8")
        plt.imshow(image, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=alpha)
        string = "GT: {}   PRED: {} ({:.2f}%)".format(gt_id, pred_id, 100 * prob)
        plt.text(50, 50, string, fontsize=12, color='white',
                 bbox=dict(fill='black', edgecolor='white', linewidth=4, alpha=0.5))

        # TODO add colormap inside the image
        if add_colormap:
            dmin=0
            dmax=255
            sm_dose = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=dmin, vmax=dmax))
            sm_dose.set_array([])
            plt.colorbar(sm_dose, shrink=0.5, ticks=np.arange(dmin, dmax, 50), orientation='vertical', pad=0.0)

        plt.axis('off')
        return figure

    def write_heatmaps(self):

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

            # construct our gradient model by supplying (1) the inputs
            # to our pre-trained model, (2) the output of the (presumably)
            # final 4D layer in the network, and (3) the output of the
            # softmax activations from the model
            self.gradModel = Model(inputs=[self.model.inputs],
                                   outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        batch_id = 0
        for i in range(self.nb_images):
            x_batch, y = next(self.validation_data)
            print('Writing out heatmap {}'.format(i))
            preds = self.model(x_batch)
            gt_id = np.argmax(y, axis=-1)[0]
            labels = np.argmax(preds, axis=-1)[0]

            pred_id = labels[batch_id]
            prob = preds[0][batch_id][pred_id]
            gt_id = gt_id[batch_id]
            x = x_batch[0][batch_id]

            # initialize our gradient class activation map and build the heatmap
            heatmap = self.compute_heatmap(x, pred_id)
            if i == 0:
                add_colormap = True
            else:
                add_colormap = False
            figure = self.plot_heatmap(heatmap, x, gt_id=gt_id, pred_id=pred_id, prob=prob, alpha=0.5,
                                       add_colormap=add_colormap)
            image = self.plot_to_image(figure)
            if i == 0:
                out_image = image
            else:
                out_image = tf.concat([out_image, image], axis=2)

        return out_image

    def on_epoch_end(self, epoch, logs=None):
        # Log the confusion matrix as an image summary.
        if self.frequency != 0 and epoch != 0 and epoch % self.frequency == 0:
            with self.file_writer_cm.as_default():
                tf.summary.image("Heatmap", self.write_heatmaps(), step=epoch)
