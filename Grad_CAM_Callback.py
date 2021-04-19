__author__ = 'Bastien Rigaud'

import os
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

import cv2

from .heatmap_tools import *
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image

def compute_Grad_CAM(gradModel, image, classIdx, eps=1e-24):
    # record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        # inputs = tf.cast(image, tf.float32)
        # (convOutputs, predictions) = self.gradModel(inputs)
        # if not len(image.shape) > 3:
        #     image = image[None, ...]
        if image.shape[0] == 1:
            image = tf.squeeze(image, axis=0)
        (convOutputs, predictions) = gradModel(image[None, ...])
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
    weights = tf.reduce_mean(guidedGrads, axis=tuple(range(-len(guidedGrads.shape),-1)))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (image.shape[-2], image.shape[-3])
    if len(cam.shape) == 3:
        z = cam.shape[0]
        heatmap = np.zeros(shape=(z, w, h))
        for ind in range(z):
            heatmap[ind] = cv2.resize(cam.numpy()[ind], (w, h))
    else:
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

class Add_Grad_CAM(Callback):
    def __init__(self, log_dir, validation_steps, validation_data=None, class_names=[], frequency=5, nb_images=5,
                 layerName=None, image_rows=512, image_cols=512, colormap_as_contour=False):
        super(Add_Grad_CAM, self).__init__()
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
        self.colormap_as_contour = colormap_as_contour
        self.file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, 'val_Grad_CAM'))

    def write_heatmaps(self):

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = find_target_layer(self.model)

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
            heatmap = compute_Grad_CAM(self.gradModel, x, pred_id)
            if i == 0:
                add_colormap = True
            else:
                add_colormap = False

            if len(x.shape) == 4:
                # keep_indice=int(np.median(np.where(heatmap >= np.max(heatmap)), axis=-1)[0]) #median slice of slices with high activation
                keep_indice = x.shape[0]//2 #median slice
                heatmap=heatmap[keep_indice]
                x=x[keep_indice]

            figure = plot_heatmap(heatmap, x, gt_id=gt_id, pred_id=pred_id, prob=prob, class_names=self.class_names, alpha=0.5,
                                       add_colormap=add_colormap, contour=self.colormap_as_contour)
            image = plot_to_image(figure, image_rows=self.image_rows, image_cols=self.image_cols)
            if i == 0:
                out_image = image
            else:
                out_image = tf.concat([out_image, image], axis=2)

        return out_image

    def on_epoch_end(self, epoch, logs=None):
        # Log the confusion matrix as an image summary.
        if self.frequency != 0 and epoch != 0 and epoch % self.frequency == 0:
            with self.file_writer_cm.as_default():
                tf.summary.image("Grad_CAM", self.write_heatmaps(), step=epoch)
