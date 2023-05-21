import click
from sklearn.model_selection import train_test_split
from util import load_data, save_images


@click.command()
@click.option('--images_dir', default='data/images/', help='Images path')
@click.option('--distorted_dir', default='data/distorted/', help='Distorted images path')
@click.option('--train_input_dir', default='data/train_input/', help='Train input path')
@click.option('--train_target_dir', default='data/train_target/', help='Train target path')
@click.option('--test_input_dir', default='data/test_input/', help='Test input path')
@click.option('--test_target_dir', default='data/test_target/', help='Test target path')
def split_command(images_dir, distorted_dir, train_input_dir, train_target_dir, test_input_dir, test_target_dir):
    _, images = load_data(images_dir)
    _, distorteds = load_data(distorted_dir)

    X_train, X_test, y_train, y_test = train_test_split(distorteds,
                                                        images,
                                                        test_size=0.2,
                                                        shuffle=False,
                                                        random_state=1004)

    save_images([str(number) + '.jpg' for number in range(len(X_train))], X_train, train_input_dir)
    save_images([str(number) + '.jpg' for number in range(len(y_train))], y_train, train_target_dir)
    save_images([str(number) + '.jpg' for number in range(len(X_test))], X_test, test_input_dir)
    save_images([str(number) + '.jpg' for number in range(len(y_test))], y_test, test_target_dir)


if __name__ == '__main__':
    print('#################### data_splitter start ####################')
    data = split_command()
    print('#################### data_splitter end ####################')
