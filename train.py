import click
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.python.keras.models import load_model
from model import GSAFE
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow_addons.optimizers import AdamW
from util import load_data, set_gpu, custom_loss


def batch_generator(X_train, Y_train):

    while True:

        for fl, lb in zip(X_train, Y_train):
            sam, lam = fl, lb

            max_iter = sam.shape[0]
            sample = []  # store all the generated data batches
            label = []   # store all the generated label batches
            i = 0
            for d, l in zip(sam, lam):
                sample.append(d)
                label.append(l)
                i += 1
                if i == max_iter:
                    break
            sample = np.asarray(sample)
            label = np.asarray(label)

            sample = np.expand_dims(sample, axis=0)
            label = np.expand_dims(label, axis=0)

            yield sample, label


@click.command()
@click.option('--train_dir', default='data/train_input/', help='Train path')
@click.option('--target_dir', default='data/train_target/', help='Target path')
@click.option('--batch_size', default=8, help='Batch size')
@click.option('--epochs', default=300, help='Epochs')
def training(train_dir, target_dir, batch_size, epochs):
    save_name = 'my_model'

    model = GSAFE()  # paper
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=custom_loss)  # paper

    # plot_model(model, to_file='model.png')
    model.summary()

    checkpoint = ModelCheckpoint(save_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    cos_decay_ann = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=0.001, first_decay_steps=15, t_mul=1.4,
                                                              m_mul=0.9, alpha=0)
    ls = LearningRateScheduler(cos_decay_ann, verbose=1)

    callbacks_list = [checkpoint, es, ls]

    print('start load')
    _, X_train = load_data(train_dir)
    _, y_train = load_data(target_dir)

    # batch fit
    train_num = int(len(X_train) * 0.8)
    print('train_num:', train_num)

    train_generator = batch_generator(X_train[:train_num], y_train[:train_num])
    val_generator = batch_generator(X_train[train_num:], y_train[train_num:])

    steps = int(train_num / batch_size)
    print('steps:', steps)

    history = model.fit_generator(generator=train_generator, steps_per_epoch=steps, validation_data=val_generator, validation_steps=steps,
                                  epochs=epochs, callbacks=callbacks_list, workers=1)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_len = np.arange(len(train_loss))
    val_len = np.arange(len(val_loss))
    plt.plot(train_len, train_loss, marker='.', c='blue', label="Train-set Loss")
    plt.plot(val_len, val_loss, marker='.', c='red', label="Validation-set Loss")
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    print('#################### train start ####################')
    set_gpu()
    training()
    print('#################### train end ####################')
