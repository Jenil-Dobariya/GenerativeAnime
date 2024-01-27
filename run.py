import tensorflow as tf
import matplotlib.pyplot as plt
import os

output_dir_ = "generated_images"
latent_dim_ = 512
x_ = 2
y_ = 2


def generate_img(model_, input_, x, y):
    prediction = model_.predict(input_)
    fig, axes = plt.subplots(x, y, figsize=(14, 14))
    idx = 0
    for row in range(x):
        for col in range(y):
            img = prediction[idx] * 255.0
            img = img.astype("int32")
            axes[row, col].imshow(img)
            idx += 1
    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)
    plt.savefig(output_dir_ + "/img{:04d}_{:04d}.jpg".format(x, y))
    plt.close()


model = tf.keras.models.load_model('trained_model.h5')
random_vector_ = tf.random.normal(shape=(x_ * y_, latent_dim_))

generate_img(model, random_vector_, x_, y_)