# To test the model use the following code:
# python run_model.py model=<model> resolution=<resolution> use_gpu=<use_gpu>
# <model> = {iphone, blackberry, sony}
# <resolution> = {orig, high, medium, small, tiny}
# <use_gpu> = {true, false}
# example:  python run_model.py model=iphone resolution=orig use_gpu=true

from scipy import misc
import numpy as np
import tensorflow as tf
from model import resnet, PyNET
from PIL import Image
from skimage import img_as_ubyte
import utils
import os
import sys
import imageio

# process command arguments
phone, resolution, use_gpu, model = utils.process_command_args(sys.argv)
test_dir = "test_photos/" + phone + "/"
test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

for photo in test_photos:
    print("Processing image " + photo)

    # disable gpu if specified
    config = tf.ConfigProto(
        device_count={'GPU': 0}) if use_gpu == "false" else None

    # get all available image resolutions
    res_sizes = utils.get_resolutions(test_dir + photo)

    # get the specified image resolution
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(
        res_sizes, phone, resolution)

    # create placeholders for input images
    x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
    x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # generate enhanced image
    # enhanced = resnet(x_image)
    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 = PyNET(x_, instance_norm=True, instance_norm_level_1=False)
    enhanced = output_l0
    
    with tf.compat.v1.Session(config=config) as sess:

        # load pre-trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "models/" + model)

        # load training image and crop it if necessary
        image = np.float16(np.array(Image.fromarray(
            imageio.imread(test_dir + photo)))) / 255
        # image = np.float16(np.array(Image.fromarray(imageio.imread(test_dir + photo)).resize(tuple(res_sizes[phone][::-1])))) / 255
        image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
        image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

        # get enhanced image
        enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
        enhanced_image = np.reshape(
            enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

        before_after = np.hstack((image_crop, enhanced_image))
        photo_name = photo.rsplit(".", 1)[0]

        # save the results as .png images
        # imageio.imwrite("results/" + photo_name + "_original.png", image_crop)
        # imageio.imwrite("results/" + photo_name +
        #                 "_processed.png", enhanced_image)
        imageio.imwrite("results/" + photo_name + "_" + model +
                        "_before_after.png", before_after)

    tf.compat.v1.reset_default_graph()
