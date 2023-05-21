import click
import cv2
import numpy as np
import random
from util import load_data, save_images


def fisheye(names, images):
    with open('parameters.txt', 'w') as f:
        f.close()

    distorteds = images.copy()
    k1, k2, k3 = 0.2, 0.05, 0.0  # Barrel Distortion

    for i, name, image in zip(range(len(images)), names, images):
        k1 = random.uniform(0.01, 0.2)
        k2 = random.uniform(0.01, 0.2)
        k3 = 0

        try:
            rows, cols, channel = image.shape[:3]

            mapy, mapx = np.indices((rows, cols), dtype=np.float32)

            mapx = 2 * mapx / (cols - 1) - 1
            mapy = 2 * mapy / (rows - 1) - 1

            r, theta = cv2.cartToPolar(mapx, mapy)
            ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

            mapx, mapy = cv2.polarToCart(ru, theta)
            mapx = ((mapx + 1) * cols - 1) / 2
            mapy = ((mapy + 1) * rows - 1) / 2

            distored = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

            crop_row = 0
            crop_column = 0
            for j, row_axis in zip(range(len(distored)), np.sum(distored, axis=1)):
                if np.sum(row_axis) != 0:
                    crop_row = j
                    break

            for j, column_axis in zip(range(len(distored)), np.sum(distored, axis=0)):
                if np.sum(column_axis) != 0:
                    crop_column = j
                    break

            if crop_row != 0:
                crop_row -= 1
            if crop_column != 0:
                crop_column -= 1

            distorteds[i] = distored

            print('{} - {} - {} - {} \n'.format(i, k1, k2, k3))
            with open('parameters.txt', 'a+') as f:
                f.write('{} - {} - {} - {} \n'.format((i + 1), k1, k2, k3))
                f.close()
        except Exception as e:
            print('error', e)

    return distorteds


@click.command()
@click.option('--images_dir', default='data/images/', help='Images path')
@click.option('--distorted_dir', default='data/distorted/', help='Distorted images path')
def generate_command(images_dir, distorted_dir):
    names, images = load_data(images_dir)
    distorteds = fisheye(names, images)
    save_images(names, distorteds, distorted_dir)


if __name__ == '__main__':
    print('#################### data_generator start ####################')
    data = generate_command()
    print('#################### data_generator end ####################')
