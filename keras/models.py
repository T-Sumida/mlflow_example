# -*- coding:utf-8 -*-
from typing import Tuple
import keras

def get_ResNet50(target_size: Tuple[int, int], class_num: int) -> keras.Model:
    """create resnet50

    Args:
        target_size (Tuple[int, int]): target image size
        class_num (int): number of output class

    Returns:
        keras.Model: resnet50
    """
    base = keras.applications.ResNet50(
        include_top=False,
        input_shape=(target_size, target_size, 3),
        weights="imagenet"
    )
    X = base.output
    X = keras.layers.GlobalAveragePooling2D()(X)
    X = keras.layers.Dropout(0.25)(X)
    output = keras.layers.Dense(
        class_num, activation='softmax'
    )(X)

    return keras.Model(inputs=[base.input], outputs=[output])

