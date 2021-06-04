# -*- coding: utf-8 -*-

from __future__ import print_function

import random
from os import path
from sys import modules

from pkg_resources import resource_filename
from tensorflow.python.keras.optimizers import SGD

from mnet_deep_cdr import Model_MNet as DeepModel
from mnet_deep_cdr.mnet_utils import dice_coef_loss, train_loader, mk_dir, files_with_ext

parent_dir = path.dirname(resource_filename(modules[__name__].__name__, '__init__.py'))
result_path = mk_dir(path.join(parent_dir, 'deep_model'))
pre_model_file = path.join(parent_dir, 'deep_model', 'Model_MNet_REFUGE.h5')
save_model_file = path.join(parent_dir, 'deep_model', 'Model_MNet_REFUGE_v2.h5')

root_path = path.join(parent_dir, 'training_crop')
train_data_path = path.join(root_path, 'data')
train_mask_path = path.join(root_path, 'label')

val_data_path = path.join(root_path, 'val_data', 'data')
val_mask_path = path.join(root_path, 'val_data', 'label')

# load training data
train_list = files_with_ext(train_data_path, '.png')
val_list = files_with_ext(val_data_path, '.png')

Total_iter = 100
nb_epoch_setting = 3
input_size = 400
optimizer_setting = SGD(lr=0.0001, momentum=0.9)

my_model = DeepModel.DeepModel(size_set=input_size)
my_model.load_weights(pre_model_file, by_name=True)

my_model.compile(optimizer=optimizer_setting, loss=dice_coef_loss, loss_weights=[0.1, 0.1, 0.1, 0.1, 0.6])

loss_max = 10000

for idx_iter in range(Total_iter):
    random.shuffle(train_list)
    model_return = my_model.fit_generator(
        generator=train_loader(train_list, train_data_path, train_mask_path, input_size),
        steps_per_epoch=len(train_list),
        validation_data=train_loader(val_list, val_data_path, val_mask_path, input_size),
        validation_steps=len(train_list),
        verbose=0
    )
    val_loss = model_return.history['val_loss'][0]
    train_loss = model_return.history['loss'][0]
    if val_loss < loss_max:
        my_model.save(save_model_file)
        loss_max = val_loss
        print('[Save] training iter: {idx}, train_loss: {train_loss}, val_loss: {val_loss}'.format(
            idx=idx_iter + 1, train_loss=train_loss, val_loss=val_loss)
        )
    else:
        print('[None] training iter: {idx}, train_loss: {train_loss}, val_loss: {val_loss}'.format(
            idx=idx_iter + 1, train_loss=train_loss, val_loss=val_loss)
        )
