import click
from tensorflow.python.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from util import load_data, save_images, set_gpu, custom_loss


def validate(predicted, y_test):
    with open('validate.txt', 'w') as f:
        f.close()

    psnr_list = []
    ssim_list = []

    for i, y, p in zip(range(len(y_test)), y_test, predicted):
        psnr = peak_signal_noise_ratio(y, p)
        ssim = structural_similarity(y, p, multichannel=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        with open('validate.txt', 'a+') as f:
            f.write('{} - {} / {} \n'.format(i, psnr, ssim))
            f.close()

    psnr_average = sum(psnr_list) / len(psnr_list)
    ssim_average = sum(ssim_list) / len(ssim_list)
    psnr_max = max(psnr_list)
    psnr_min = min(psnr_list)
    ssim_max = max(ssim_list)
    ssim_min = min(ssim_list)

    with open('validate.txt', 'a+') as f:
        f.write('psnr, ssim average {} / {} \n'.format(psnr_average, ssim_average))
        f.write('psnr, ssim max {} / {} \n'.format(psnr_max, ssim_max))
        f.write('psnr, ssim min {} / {} \n'.format(psnr_min, ssim_min))
        f.close()

    print('psnr average:', psnr_average)
    print('ssim average:', ssim_average)
    print('psnr_list max min:', psnr_max, psnr_min)
    print('ssim_list max min:', ssim_max, ssim_min)


@click.command()
@click.option('--test_dir', default='data/test_input/', help='Test path')
@click.option('--target_dir', default='data/test_target/', help='Target path')
@click.option('--predicted_dir', default='predicted/test_output/', help='Predicted path')
@click.option('--batch_size', default=1, help='Batch size')
def testing(test_dir, target_dir, predicted_dir, batch_size):
    leaky_relu = tf.nn.leaky_relu

    target_names, X_test = load_data(test_dir)

    loaded_model = load_model(save_name + '.h5', custom_objects={'custom_loss': custom_loss, 'leaky_relu': leaky_relu})
    loaded_model.summary()

    n = 0
    step = 1000

    for i in range(10):
        print('start end', n, (n + step))
        predicted = loaded_model.predict(X_test[n:n + step], batch_size=batch_size, verbose=1)

        save_images(target_names[n:n + step], predicted, predicted_dir)

        n = n + step

    target_names, target_images = load_data(target_dir)
    predict_names, predict_images = load_data(predicted_dir)

    for t_name, p_name, t_image, p_image in zip(target_names, predict_names, target_images, predict_images):
        print('t_name, p_name:', t_name, p_name)

    validate(predict_images, target_images)


if __name__ == '__main__':
    print('#################### test ####################')
    set_gpu()
    testing()