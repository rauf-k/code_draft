from lib.batch_manager import BatchManager
from lib.batch_processor import BatchProcessor
from lib.inference_test import InferenceManager
from mod import model1
import utils.save_history as sh
import tensorflow as tf
from const import data_inf_prod_consts as DC
from lib.numtry import NumTry
from tensorflow import keras
import numpy as np
import random
from utils import my_sql_connector as sql
nt = NumTry()


def get_sub_batch(sub_batch_size, bm, bp):
    x_p_list = []
    x_s_list = []
    x_t_list = []

    y_list = []

    for sample in range(sub_batch_size):
        training_tables = bm.get_training_table_names()
        random_table = random.choice(training_tables)
        random_start_row = random.randrange(1, random_table.get_number_of_rows() - (DC.PAST_LENGTH + DC.FUTURE_LENGTH + 1))
        raw_sample = sql.get_specific_rows_from_table(random_table.get_table_name(), random_start_row, random_start_row + (DC.PAST_LENGTH + DC.FUTURE_LENGTH + 1))
        x_p, x_s, x_t, y = bp.process_batch_with_1_sample_2d(raw_sample)

        x_p_list.append(x_p)
        x_s_list.append(x_s)
        x_t_list.append(x_t)
        y_list.append(y)

    return np.array(x_p_list).astype('float32'), np.array(x_s_list).astype('float32'), np.array(x_t_list).astype(np.int64), np.array(y_list).astype('float32')


def train_with_ga():
    model = model1.ModelCategorical(DC.NUMBER_OF_SAMPLES_IN_A_SUB_BATCH)
    # model.load_weights('res/***.h5')  # if continuing to train saved model
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam")
    model.summary()

    bm = BatchManager()
    print('bm done')
    bp = BatchProcessor()
    print('bp done')
    im = InferenceManager(bm)
    print('im done')

    gradients_to_collect = int(DC.NUMBER_OF_SAMPLES_IN_A_FULL_BATCH / DC.NUMBER_OF_SAMPLES_IN_A_SUB_BATCH)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model_update_counter = 0
    while True:
        total_loss = 0
        train_vars = model.trainable_variables
        accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]

        for gr_ in range(gradients_to_collect):
            x_p, x_s, x_t, y = get_sub_batch(DC.NUMBER_OF_SAMPLES_IN_A_SUB_BATCH, bm, bp)
            with tf.GradientTape() as tape:
                logits = model([x_p, x_s, x_t], training=True)
                loss_value = keras.losses.mean_squared_error(y, logits)

            total_loss += loss_value
            gradients = tape.gradient(loss_value, train_vars)
            accum_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_gradient, gradients)]

        accum_gradient = [this_grad / gradients_to_collect for this_grad in accum_gradient]
        optimizer.apply_gradients(zip(accum_gradient, train_vars))

        x_p_, x_s_, x_t_, y_ = get_sub_batch(20, bm, bp)
        eval = model.evaluate([x_p_, x_s_, x_t_], y_, verbose=1)
        msg = "model_update_num: {} loss: {} eval: {}".format(model_update_counter, total_loss / gradients_to_collect, eval)
        sh.save_history(msg)
        print(msg)

        if model_update_counter >= DC.INFERENCE_AFTER_Nth_BATCH - 1 and model_update_counter % DC.INFERENCE_EVERY_Nth_BATCH == 0:
            im.perform_inference(model, model_update_counter)

        model_update_counter += 1


def inference_only():
    bm = BatchManager()
    print('bm done')
    bp = BatchProcessor()
    print('bp done')
    im = InferenceManager(bm)
    print('im done')

    list_of_models_for_inference = [
        '***.h5',
    ]

    for model_name in list_of_models_for_inference:
        model = model1.ModelCategorical(DC.NUMBER_OF_SAMPLES_IN_A_SUB_BATCH)
        model.load_weights('res/' + model_name)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam")
        model.summary()
        im.perform_inference(model, 0, model_name)


if __name__ == '__main__':
    train_with_ga()
    # inference_only()

