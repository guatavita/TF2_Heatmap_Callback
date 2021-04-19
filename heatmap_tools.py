import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

def find_target_layer(model):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) >= 4 and ('activation' in layer.name.lower() or 'pooling' not in layer.name.lower()):
            return layer.name
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

def plot_heatmap(heatmap, image, gt_id, pred_id, prob, class_names=None, alpha=0.5, eps=1e-8, add_colormap=False,
                 contour=False, cmap = 'jet'):
    '''
    :param heatmap:
    :param image:
    :param gt_id:
    :param pred_id:
    :param prob:
    :param class_names:
    :param alpha:
    :param eps:
    :param add_colormap:
    :param contour:
    :param cmap: jet or inferno
    :return:
    '''

    figure = plt.figure()
    image = image.numpy()
    numer = image - np.min(image)
    denom = (image.max() - image.min()) + eps
    image = numer / denom
    image = (image * 255).astype("uint8")
    plt.imshow(image, cmap='gray')
    if contour:
        plt.contour(heatmap, cmap=cmap, levels=10, alpha=1.0)
    else:
        plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    if class_names:
        string = "GT: {}   PRED: {} ({:.2f}%)".format(class_names[gt_id], class_names[pred_id], 100 * prob)
    else:
        string = "GT: {}   PRED: {} ({:.2f}%)".format(gt_id, pred_id, 100 * prob)
    plt.text(50, 50, string, fontsize=12, color='white',
             bbox=dict(fill='black', edgecolor='white', linewidth=4, alpha=0.5))

    # TODO add colormap inside the image
    if add_colormap:
        dmin=0
        dmax=255
        sm_dose = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=dmin, vmax=dmax))
        sm_dose.set_array([])
        plt.colorbar(sm_dose, shrink=0.5, ticks=np.arange(dmin, dmax, 50), orientation='vertical', pad=0.0)

    plt.axis('off')
    return figure

def plot_to_image(figure, image_rows, image_cols):
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
    image = tf.image.resize(image, (image_rows, image_cols), method='bilinear',
                            preserve_aspect_ratio=True)
    image = tf.image.resize_with_crop_or_pad(image, target_height=image_rows, target_width=image_cols)
    image = tf.cast(image, np.uint8)
    return image


def generate_path_inputs(baseline, input, alphas):
    """Generate m interpolated inputs between baseline and input features.
    Args:
      baseline(Tensor): A 3D image tensor of floats with the shape
        (img_height, img_width, 3).
      input(Tensor): A 3D image tensor of floats with the shape
        (img_height, img_width, 3).
      alphas(Tensor): A 1D tensor of uniformly spaced floats with the shape
        (m_steps,).
    Returns:
      path_inputs(Tensor): A 4D tensor of floats with the shape
        (m_steps, img_height, img_width, 3).
    """
    # Expand dimensions for vectorized computation of interpolations.
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(input, axis=0)
    delta = input_x - baseline_x
    path_inputs = baseline_x + alphas_x * delta

    return path_inputs
