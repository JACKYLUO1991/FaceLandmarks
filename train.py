import argparse
# from model import PFLDNetBackbone
from model import MobileNetV3
from data_generator import DataGenerator
from loss import *
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
import os
import numpy as np

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

ap = argparse.ArgumentParser()
ap.add_argument("--batch_size", type=int, default=128,
                help="batch size of data")
# ap.add_argument("--alpha", type=float, default=1.0,
#                 help="control width parameter of of MobileNet blocks")
ap.add_argument("--lr", type=float, default=1e-3,
                help="learning rate")
ap.add_argument("--checkpoints", type=str, default="./checkpoints/pfld.h5",
                help="checkpoint path")
ap.add_argument("--fine_tune_path", type=str, default="./checkpoints/pfld.h5",
                help="fine tune checkpoint path")
ap.add_argument('--fine_tune', action='store_true', help='fine tune or not')
ap.add_argument('--epochs', type=int,
                default=100, help='epoch of training')
ap.add_argument('--workers', type=int,
                default=4, help='how many workers')
args = vars(ap.parse_args())


class PolyDecay:
    '''
    Exponential decay strategy implementation
    '''

    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs

    def scheduler(self, epoch):
        return self.initial_lr * np.power(1.0 - 1.0 * epoch / self.n_epochs, self.power)


if __name__ == '__main__':

    # Set GPU variable
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    train_generator = DataGenerator(
        batch_size=args['batch_size'], root_dir='./new_dataset', csv_file='./new_dataset/face_mixed.csv',
        shuffle=True, transformer=True)
    val_generator = DataGenerator(
        batch_size=args['batch_size'], root_dir='./new_test_dataset', csv_file='./new_test_dataset/face_mixed.csv')

    # model = PFLDNetBackbone(input_shape=(112, 112, 3),
    #                         output_nodes=212, alpha=args['alpha'])
    model = MobileNetV3(shape=(112, 112, 3), n_class=212).build()

    if args['fine_tune']:
        model.load_weights(args['fine_tune_path'], by_name=True)

    # https://blog.csdn.net/laolu1573/article/details/83626555
    # we can samply set 'b2_s' in loss_weights to 0...
    model.compile(loss={'b1_s': wing_loss, 'b2_s': smoothL1}, loss_weights={'b1_s': 2, 'b2_s': 1},
                  optimizer=Adam(lr=args['lr']),
                  metrics={'b1_s': normalized_mean_error, 'b2_s': 'mae'})

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    filepath = "./checkpoints/{epoch:02d}-{val_loss:.5f}.h5"
    tensorboard = callbacks.TensorBoard(log_dir='./checkpoints/logs')
    checkpoint = callbacks.ModelCheckpoint(
        filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lr_decay = callbacks.LearningRateScheduler(
        PolyDecay(args['lr'], 0.9, args['epochs']).scheduler)
    callbacks_list = [checkpoint, tensorboard, lr_decay]

    model.fit_generator(
        train_generator,
        len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=args['epochs'],
        verbose=1,
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=args['workers']
    )
    model.save(args['checkpoints'])

    K.clear_session()
