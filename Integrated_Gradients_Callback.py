__author__ = 'Bastien Rigaud'

# from https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/integrated_gradients/integrated_gradients.ipynb

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

import cv2

from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image

from .heatmap_tools import *


class Add_Integrated_Gradients(Callback):
    def __init__(self, log_dir, validation_steps, validation_data=None, class_names=[], frequency=5, nb_images=5,
                 layerName=None, image_rows=512, image_cols=512, nb_channels=3, colormap_as_contour=False,
                 alpha_steps=50):
        super(Add_Integrated_Gradients, self).__init__()
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
        self.nb_channels = nb_channels
        self.colormap_as_contour = colormap_as_contour
        self.alpha_steps = alpha_steps
        self.name_baseline_tensors = {
            'Black': tf.zeros(shape=(image_rows, image_cols, nb_channels)),
            'Random': tf.random.uniform(shape=(image_rows, image_cols, nb_channels), minval=0.0, maxval=1.0),
            'White': tf.ones(shape=(image_rows, image_cols, nb_channels)),
        }
        self.file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, 'val_Int_Grads'))

    def compute_gradients(self, model, path_inputs, target_class_idx):
        """Compute gradients of model predicted probabilties with respect to inputs.
        Args:
          mode(tf.keras.Model): Trained Keras model.
          path_inputs(Tensor): A 4D tensor of floats with the shape
            (m_steps, img_height, img_width, 3).
          target_class_idx(Tensor): A 0D tensor of an int corresponding to the correct
            ImageNet target class index.
        Returns:
          gradients(Tensor): A 4D tensor of floats with the shape
            (m_steps, img_height, img_width, 3).
        """
        with tf.GradientTape() as tape:
            tape.watch(path_inputs)
            predictions = model(path_inputs)
            # Note: IG requires softmax probabilities; converting Inception V1 logits.
            # outputs = tf.nn.softmax(predictions, axis=-1)[:, target_class_idx]
            outputs = predictions[0][:, target_class_idx]
        gradients = tape.gradient(outputs, path_inputs)

        return gradients

    def integral_approximation(self, gradients, method='riemann_trapezoidal'):
        """Compute numerical approximation of integral from gradients.

        Args:
          gradients(Tensor): A 4D tensor of floats with the shape
            (m_steps, img_height, img_width, 3).
          method(str): A string representing the integral approximation method. The
            following methods are implemented:
            - riemann_trapezoidal(default)
            - riemann_left
            - riemann_midpoint
            - riemann_right
        Returns:
          integrated_gradients(Tensor): A 3D tensor of floats with the shape
            (img_height, img_width, 3).
        """
        if method == 'riemann_trapezoidal':
            grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        elif method == 'riemann_left':
            grads = gradients
        elif method == 'riemann_midpoint':
            grads = gradients
        elif method == 'riemann_right':
            grads = gradients
        else:
            raise AssertionError("Provided Riemann approximation method is not valid.")

        # Average integration approximation.
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)

        return integrated_gradients

    def generate_alphas(self, m_steps=50,
                        method='riemann_trapezoidal'):
        """
        Args:
          m_steps(Tensor): A 0D tensor of an int corresponding to the number of linear
            interpolation steps for computing an approximate integral. Default is 50.
          method(str): A string representing the integral approximation method. The
            following methods are implemented:
            - riemann_trapezoidal(default)
            - riemann_left
            - riemann_midpoint
            - riemann_right
        Returns:
          alphas(Tensor): A 1D tensor of uniformly spaced floats with the shape
            (m_steps,).
        """
        m_steps_float = tf.cast(m_steps, float)  # cast to float for division operations.

        if method == 'riemann_trapezoidal':
            alphas = tf.linspace(0.0, 1.0, m_steps + 1)  # needed to make m_steps intervals.
        elif method == 'riemann_left':
            alphas = tf.linspace(0.0, 1.0 - (1.0 / m_steps_float), m_steps)
        elif method == 'riemann_midpoint':
            alphas = tf.linspace(1.0 / (2.0 * m_steps_float), 1.0 - 1.0 / (2.0 * m_steps_float), m_steps)
        elif method == 'riemann_right':
            alphas = tf.linspace(1.0 / m_steps_float, 1.0, m_steps)
        else:
            raise AssertionError("Provided Riemann approximation method is not valid.")

        return alphas

    def integrated_gradients(self, model,
                             baseline,
                             input,
                             target_class_idx,
                             m_steps=50,
                             method='riemann_trapezoidal',
                             batch_size=32
                             ):
        """
        Args:
          model(keras.Model): A trained model to generate predictions and inspect.
          baseline(Tensor): A 3D image tensor with the shape
            (image_height, image_width, 3) with the same shape as the input tensor.
          input(Tensor): A 3D image tensor with the shape
            (image_height, image_width, 3).
          target_class_idx(Tensor): An integer that corresponds to the correct
            ImageNet class index in the model's output predictions tensor. Default
              value is 50 steps.
          m_steps(Tensor): A 0D tensor of an integer corresponding to the number of
            linear interpolation steps for computing an approximate integral.
          method(str): A string representing the integral approximation method. The
            following methods are implemented:
            - riemann_trapezoidal(default)
            - riemann_left
            - riemann_midpoint
            - riemann_right
          batch_size(Tensor): A 0D tensor of an integer corresponding to a batch
            size for alpha to scale computation and prevent OOM errors. Note: needs to
            be tf.int64 and shoud be < m_steps. Default value is 32.
        Returns:
          integrated_gradients(Tensor): A 3D tensor of floats with the same
            shape as the input tensor (image_height, image_width, 3).
        """

        # 1. Generate alphas.
        alphas = self.generate_alphas(m_steps=m_steps, method=method)

        # Initialize TensorArray outside loop to collect gradients. Note: this data structure
        # is similar to a Python list but more performant and supports backpropogation.
        # See https://www.tensorflow.org/api_docs/python/tf/TensorArray for additional details.
        gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        # Note: this implementation opted for lightweight tf.range iteration with @tf.function.
        # Alternatively, you could also use tf.data, which adds performance overhead for the IG
        # algorithm but provides more functionality for working with tensors and image data pipelines.
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]

            # 2. Generate interpolated inputs between baseline and input.
            interpolated_path_input_batch = generate_path_inputs(baseline=baseline, input=input, alphas=alpha_batch)

            # 3. Compute gradients between model outputs and interpolated inputs.
            gradient_batch = self.compute_gradients(model=model, path_inputs=interpolated_path_input_batch,
                                                    target_class_idx=target_class_idx)

            # Write batch indices and gradients to TensorArray. Note: writing batch indices with
            # scatter() allows for uneven batch sizes. Note: this operation is similar to a Python list extend().
            # See https://www.tensorflow.org/api_docs/python/tf/TensorArray#scatter for additional details.
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

            # Stack path gradients together row-wise into single tensor.
        total_gradients = gradient_batches.stack()

        # 4. Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(gradients=total_gradients, method=method)

        # 5. Scale integrated gradients with respect to input.
        integrated_gradients = (input - baseline) * avg_gradients

        return integrated_gradients

    def write_heatmaps(self):

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

            ig_attributions = self.integrated_gradients(model=self.model,
                                                baseline=self.name_baseline_tensors['Black'],
                                                input=tf.cast(x, dtype=tf.float32),
                                                target_class_idx=pred_id,
                                                m_steps=self.alpha_steps,
                                                method='riemann_trapezoidal')

            attribution_mask = tf.reduce_sum(tf.math.abs(ig_attributions), axis=-1)
            attribution_mask = attribution_mask.numpy()
            numer = attribution_mask - np.min(attribution_mask)
            denom = (attribution_mask.max() - attribution_mask.min()) + 1e-24
            attribution_mask = numer / denom
            attribution_mask = (attribution_mask * 255).astype("uint8")

            if i == 0:
                add_colormap = True
            else:
                add_colormap = False

            if len(x.shape) == 4:
                # keep_indice=int(np.median(np.where(heatmap >= np.max(heatmap)), axis=-1)[0]) #median slice of slices with high activation
                keep_indice = x.shape[0] // 2  # median slice
                attribution_mask = attribution_mask[keep_indice]
                x = x[keep_indice]

            figure = plot_heatmap(attribution_mask, x, gt_id=gt_id, pred_id=pred_id, prob=prob, class_names=self.class_names,
                                  alpha=0.5, add_colormap=add_colormap, contour=self.colormap_as_contour, cmap='inferno')
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
                tf.summary.image("Int_Grads", self.write_heatmaps(), step=epoch)
