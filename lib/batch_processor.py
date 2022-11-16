import numpy as np
import const.data_inf_prod_consts as DC
from lib.numtry import NumTry


class BatchProcessor:
    def __init__(self):
        self.nt = NumTry()

    def process_batch_with_1_sample_2d(self, batch_1d):
        price_full, size_full, time_full = self.nt.separate_price_size_time_ib_1d(batch_1d, DC.IB_PRICE_INDEX, DC.IB_SIZE_INDEX, DC.IB_TIME_INDEX)
        price_past, size_past, time_past, label_bs = self.process_batch_with_1_sample_2d_helper(price_full, size_full, time_full)

        return np.array(price_past).astype('float32'), np.array(size_past).astype('float32'), np.array(time_past).astype(np.int64), np.array(label_bs).astype('float32')

    def process_batch_with_1_sample_2d_helper(self, price_full, size_full, time_full):
        price_past = price_full[:-DC.FUTURE_LENGTH]
        size_past = size_full[:-DC.FUTURE_LENGTH]
        time_past = time_full[:-DC.FUTURE_LENGTH]

        label = self.nt.label_regression_sharp_1samp_2out(price_full,
                                                          DC.FUTURE_LENGTH,
                                                          True,
                                                          DC.LABEL_PRICE_SMA_FOR_Y_CALCULATION,
                                                          DC.LABEL_2D_MA_INTERVAL,
                                                          DC.LABEL_MULTIPLIER)
        label_b = label[0]
        label_s = label[1]
        label_bs = label_b - label_s

        return price_past, size_past, time_past, label_bs

