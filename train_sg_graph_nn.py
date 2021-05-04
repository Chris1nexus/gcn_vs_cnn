from tqdm import tqdm
from stellargraph.mapper import PaddedGraphGenerator
import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph
# 
from stellargraph import datasets

from sklearn import model_selection


from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping


def create_graph_classification_model(generator, hidden_layer_sizes=(64, 64, 64, 64),
                                                  activations=('relu','relu','relu','relu'),
                                                  dropout=0.2,
                                                  learning_rate=0.001):
    '''
    generates stellargraph graph neural network
    Args:
        -generator (keras generator)
        -hidden_layer_sizes=(64, 64, 64, 64)
        -activations=('relu','relu','relu','relu'),
        -dropout=0.2,
        -learning_rate=0.001
    Returns:
        -stellargraph model
    '''
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=hidden_layer_sizes,
        activations=activations,
        generator=generator,
        dropout=dropout,
    )
    x_inp, x_out = gc_model.in_out_tensors()

    #predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=1, activation="sigmoid")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate), loss=binary_crossentropy, metrics=["acc"])

    return model

