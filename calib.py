import click
from tensorflow.python.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from util import load_data, save_images, set_gpu, custom_loss


@click.command()
@click.option('--my_model', default='my_model.h5', help='Model path')
@click.option('--sample_dir', default='sample/', help='Sample path')
@click.option('--results_dir', default='results/', help='Results path')
@click.option('--batch_size', default=4, help='Batch size')
def testing(my_model, sample_dir, results_dir, batch_size):
    names, images = load_data(sample_dir)

    loaded_model = load_model(my_model, custom_objects={'custom_loss': custom_loss})
    predicted = loaded_model.predict(images, batch_size=batch_size, verbose=1)

    save_images([name + '.jpg' for name in names], predicted, results_dir)


if __name__ == '__main__':
    print('#################### calibrate start ####################')
    set_gpu()
    testing()
    print('#################### calibrate end ####################')