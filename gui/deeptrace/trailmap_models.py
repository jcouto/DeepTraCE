# THIS IS FROM TRAILMAP MODELS;
# It is included here to make things easier to run but has not been adapted.
# The original and complete TRAILMAP code was written at Liqun Luo's lab by Albert Pun and Drew Friedmann
#
# MIT License
#
# Copyright (c) 2019 AlbertPun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.keras import backend as K
# set the log to check which devices get allocated...remove this later
tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)

import numpy as np

input_dim = 64
output_dim = 36

def create_weighted_binary_crossentropy(axon_weight, background_weight, artifact_weight, edge_weight):
    def weighted_binary_crossentropy(y_true, y_pred):
        weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)

        mask = tf.equal(weights, 1)

        axon_true = y_true[:, :, :, :, 0]
        axon_true = tf.expand_dims(axon_true, -1)
        axon_mask = tf.boolean_mask(axon_true, mask)

        background_true = y_true[:, :, :, :, 1]
        background_true = tf.expand_dims(background_true, -1)
        background_mask = tf.boolean_mask(background_true, mask)

        artifact_true = y_true[:, :, :, :, 2]
        artifact_true = tf.expand_dims(artifact_true, -1)
        artifact_mask = tf.boolean_mask(artifact_true, mask)

        edge_true = y_true[:, :, :, :, 3]
        edge_true = tf.expand_dims(edge_true, -1)
        edge_mask = tf.boolean_mask(edge_true, mask)

        mask_true = tf.boolean_mask(axon_true, mask)
        mask_pred = tf.boolean_mask(y_pred, mask)

        crossentropy = K.binary_crossentropy(mask_true, mask_pred)

        weight_vector = (axon_mask * axon_weight) + (background_mask * background_weight) + \
                        (artifact_mask * artifact_weight) + (edge_mask * edge_weight)

        weighted_crossentropy = weight_vector * crossentropy

        return K.mean(weighted_crossentropy)

    return weighted_binary_crossentropy


def weighted_binary_crossentropy(y_true, y_pred):
    loss = create_weighted_binary_crossentropy(1.5, 0.2, 0.8, 0.05)(y_true, y_pred)
    return loss


def adjusted_accuracy(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)

    mask = K.equal(weights, 1)

    axons_true = y_true[:, :, :, :, 0]
    axons_true = K.expand_dims(axons_true, -1)

    mask_true = tf.boolean_mask(axons_true, mask)
    mask_pred = tf.boolean_mask(y_pred, mask)

    return K.mean(K.equal(mask_true, K.round(mask_pred)))


def axon_precision(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.boolean_mask(y_true[:, :, :, :, 0], mask)
    mask_pred = tf.boolean_mask(y_pred[:, :, :, :, 0], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision


def axon_recall(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.boolean_mask(y_true[:, :, :, :, 0], mask)
    mask_pred = tf.boolean_mask(y_pred[:, :, :, :, 0], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(mask_true, 0, 1)))

    recall = true_positives / (actual_positives + K.epsilon())

    return recall


def artifact_precision(y_true, y_pred):
    weights = y_true[:, :, :, :, 2]

    mask = tf.equal(weights, 1)
    mask_true = tf.boolean_mask(y_true[:, :, :, :, 2], mask)
    mask_pred = tf.boolean_mask(1 - y_pred[:, :, :, :, 0], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def f1_score(y_true, y_pred):

    precision = axon_precision(y_true, y_pred)
    recall = axon_recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def edge_axon_precision(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.boolean_mask(y_true[:, :, :, :, 0], mask)
    mask_pred = tf.boolean_mask(y_pred[:, :, :, :, 0], mask)
    mask_edge_true = tf.boolean_mask(y_true[:, :, :, :, 3], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    edge_count = K.sum(K.round(K.clip(mask_edge_true * mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon() - edge_count)

    return precision


def get_net():
    from tensorflow.keras.layers import (Conv3D, MaxPooling3D,
                                         BatchNormalization,
                                         Conv3DTranspose,
                                         concatenate,
                                         Cropping3D, Input)

    # Level 1
    input = Input((input_dim, input_dim, input_dim, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(input)
    batch1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(batch1)
    batch1 = BatchNormalization()(conv1)

    # Level 2
    pool2 = MaxPooling3D((2, 2, 2))(batch1)
    conv2 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(pool2)
    batch2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation="relu", padding="same")(batch2)
    batch2 = BatchNormalization()(conv2)

    # Level 3
    pool3 = MaxPooling3D((2, 2, 2))(batch2)
    conv3 = Conv3D(128, (3, 3, 3), activation="relu", padding="same")(pool3)
    batch3 = BatchNormalization()(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation="relu", padding="same")(batch3)
    batch3 = BatchNormalization()(conv3)

    # Level 4
    pool4 = MaxPooling3D((2, 2, 2))(batch3)
    conv4 = Conv3D(256, (3, 3, 3), activation="relu", padding="same")(pool4)
    batch4 = BatchNormalization()(conv4)
    conv4 = Conv3D(512, (3, 3, 3), activation="relu", padding="same")(batch4)
    batch4 = BatchNormalization()(conv4)

    # Level 3
    up5 = Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding="same", activation="relu")(batch4)
    merge5 = concatenate([up5, batch3])
    conv5 = Conv3D(256, (3, 3, 3), activation="relu")(merge5)
    batch5 = BatchNormalization()(conv5)
    conv5 = Conv3D(256, (3, 3, 3), activation="relu")(batch5)
    batch5 = BatchNormalization()(conv5)

    # Level 2
    up6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), activation="relu")(batch5)
    merge6 = concatenate([up6, Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(batch2)])
    conv6 = Conv3D(128, (3, 3, 3), activation="relu")(merge6)
    batch6 = BatchNormalization()(conv6)
    conv6 = Conv3D(128, (3, 3, 3), activation="relu")(batch6)
    batch6 = BatchNormalization()(conv6)

    # Level 1
    up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same", activation="relu")(batch6)
    merge7 = concatenate([up7, Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(batch1)])
    conv7 = Conv3D(64, (3, 3, 3), activation="relu")(merge7)
    batch7 = BatchNormalization()(conv7)
    conv7 = Conv3D(64, (3, 3, 3), activation="relu")(batch7)
    batch7 = BatchNormalization()(conv7)

    # Output dim is (36, 36, 36)
    preds = Conv3D(1, (1, 1, 1), activation="sigmoid")(batch7)
    from tensorflow.keras.models import Model

    model = Model(inputs=input, outputs=preds)
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(lr=0.001, decay=0.00), loss=weighted_binary_crossentropy,
                  metrics=[axon_precision, axon_recall, f1_score, artifact_precision, edge_axon_precision, adjusted_accuracy])

    return model


def trailmap_apply_model( model, section,threshold = 0.01, batch_size = 16):
    '''
    This function is adapted from TRAILMAP inference.
       - model: load with get_net
       - section: a section of the array to evaluate (input_dims x H x W)
       - batch_size: change if you have more GPU memory, decrease if you don't have enough (default 16)
       - threshold: sections with less than this value are skipped (default 0.01)
    
    '''
    # List of bottom left corner coordinate of all input chunks in the section
    coords = []
    dim_offset = (input_dim - output_dim) // 2
    # Pad the section to account for the output dimension being smaller the input dimension
    # temp_section = np.pad(section, ((0, 0), (dim_offset, dim_offset),
    #                                 (dim_offset, dim_offset)), 'constant', constant_values=(0, 0))
    temp_section = np.pad(section, ((0, 0), (dim_offset, dim_offset),
                                    (dim_offset, dim_offset)), 'edge')
    # Add most chunks aligned with top left corner
    for x in range(0, temp_section.shape[1] - input_dim, output_dim):
        for y in range(0, temp_section.shape[2] - input_dim, output_dim):
            coords.append((0, x, y))
    # Add bottom side aligned with bottom side
    for x in range(0, temp_section.shape[1] - input_dim, output_dim):
        coords.append((0, x, temp_section.shape[2]-input_dim))
    # Add right side aligned with right side
    for y in range(0, temp_section.shape[2] - input_dim, output_dim):
        coords.append((0, temp_section.shape[1]-input_dim, y))
    # Add bottom right corner
    coords.append((0, temp_section.shape[1]-input_dim, temp_section.shape[2]-input_dim))
    coords = np.array(coords)
    # List of cropped volumes that the network will process
    batch_crops = np.zeros((batch_size, input_dim, input_dim, input_dim))
    # List of coordinates associated with each cropped volume
    batch_coords = np.zeros((batch_size, 3), dtype="int")
    # Keeps track of which coord we are at
    i = 0
    # Generate dummy segmentation
    seg = np.zeros(temp_section.shape).astype('float32')
    # Loop through each possible coordinate
    while i < len(coords):
        # Fill up the batch by skipping chunks below the threshold
        batch_count = 0
        while i < len(coords) and batch_count < batch_size:
            (z, x, y) = coords[i]
            # Get the chunk associated with the coordinate
            test_crop = temp_section[z:z + input_dim, x:x + input_dim, y:y + input_dim]
            # Only add chunk to batch if its max value is above threshold
            # (Avoid wasting time processing background chunks)
            if np.max(test_crop) > threshold:
                batch_coords[batch_count] = (z, x, y)
                batch_crops[batch_count] = test_crop
                batch_count += 1
            i += 1
        # Once the batch is filled up run the network on the chunks
        batch_input = np.reshape(batch_crops, batch_crops.shape + (1,))
        output = np.squeeze(model.predict(batch_input)[:, :, :, :, [0]])
        # Place the predictions in the segmentation
        for j in range(len(batch_coords)):
            (z, x, y) = batch_coords[j] + dim_offset
            seg[z:z + output_dim, x:x + output_dim, y:y + output_dim] = output[j]
    cropped_seg = seg[:, dim_offset: dim_offset + section.shape[1], dim_offset: dim_offset + section.shape[2]]
    return cropped_seg
