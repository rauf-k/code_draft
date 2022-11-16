import numpy as np
import tensorflow as tf
from mod import model1
from const import data_inf_prod_consts as DC

model_amzn = model1.ModelCategorical(DC.PROD_BATCH)
model_amzn.load_weights(DC.PROD_MODEL_PATH_NAME)
model_amzn.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam")
model_amzn.summary()


def infer(input_batch):
    pred = model_amzn(input_batch, training=False)
    max_val = np.max(pred)
    min_val = np.min(pred)

    return max_val, min_val
