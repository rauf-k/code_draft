import time
import numpy as np

class RunningListAvg:
    def __init__(self, max_len, max_percent_spike):
        self.max_len = max_len
        self.max_percent_spike = float(max_percent_spike)
        self.list_full = False
        self.list_of_values = []

    def add_value(self, value):
        self.list_of_values.append(value)
        if len(self.list_of_values) > self.max_len:
            self.list_of_values.pop(0)
            self.list_full = True

    def get_list(self):
        return self.list_full, self.list_of_values

    def _remove_spikes(self, price_seq, max_percent_spike):
        tmp_prev = price_seq[0]
        no_spikes_price = []
        for k in range(1, len(price_seq)):
            percent_diff = abs(((price_seq[k] - price_seq[k - 1]) / price_seq[k - 1]) * 100.0)
            if percent_diff >= max_percent_spike:
                no_spikes_price.append(tmp_prev)
            else:
                no_spikes_price.append(price_seq[k])
                tmp_prev = price_seq[k]

        no_spikes_price = no_spikes_price + [no_spikes_price[-1]]

        return no_spikes_price

    def get_average(self):
        if len(self.list_of_values) > 1:
            clean_values = self._remove_spikes(self.list_of_values, self.max_percent_spike)
            return np.average(clean_values)
        elif len(self.list_of_values) == 1:
            return np.average(self.list_of_values)
        elif len(self.list_of_values) == 0:
            raise ValueError('bad')


class RunningListPstConst:
    def __init__(self, max_len):
        self.max_len = max_len
        self.list_full = False
        self.list_of_price = []
        self.list_of_size = []
        self.list_of_time = []

    def add_value(self, price, size, time):
        self.list_of_price.append(price)
        self.list_of_size.append(size)
        self.list_of_time.append(time)
        if len(self.list_of_price) > self.max_len:
            self.list_of_price.pop(0)
            self.list_of_size.pop(0)
            self.list_of_time.pop(0)
            self.list_full = True

    def get_list(self):
        return self.list_full, self.list_of_price, self.list_of_size, self.list_of_time


class RunningListPstBuff:
    def __init__(self, max_len):
        self.max_len = max_len
        self.list_full = False
        self.list_of_price = []
        self.list_of_size = []
        self.list_of_time = []

    def add_value(self, price, size, time):
        self.list_of_price.append(price)
        self.list_of_size.append(size)
        self.list_of_time.append(time)

    def get_list(self):
        if len(self.list_of_price) > self.max_len:
            self.list_of_price.pop(0)
            self.list_of_size.pop(0)
            self.list_of_time.pop(0)
            self.list_full = True
        return self.list_full, self.list_of_price, self.list_of_size, self.list_of_time


class RunningListPstBuffBlowOff:
    def __init__(self, max_len, blow_off_limit=30):
        self.max_len = max_len
        self.blow_off_limit = blow_off_limit
        self.list_full = False
        self.list_of_price = []
        self.list_of_size = []
        self.list_of_time = []

    def add_value(self, price, size, time):
        self.list_of_price.append(price)
        self.list_of_size.append(size)
        self.list_of_time.append(time)

    def get_list(self):
        seq_len = len(self.list_of_price)
        overflow_val = seq_len - self.max_len
        if overflow_val > self.blow_off_limit:
            self.list_of_price = self.list_of_price[-self.max_len]
            self.list_of_size = self.list_of_size[-self.max_len]
            self.list_of_time = self.list_of_time[-self.max_len]
            self.list_full = True
            return self.list_full, self.list_of_price, self.list_of_size, self.list_of_time

        if seq_len > self.max_len:
            self.list_of_price.pop(0)
            self.list_of_size.pop(0)
            self.list_of_time.pop(0)
            self.list_full = True
        return self.list_full, self.list_of_price, self.list_of_size, self.list_of_time


class RunningListPstBatch:
    """
    use case:

    rl = RunningListPstBatch(seq_len=10, batch_size=5)
    rl.add(values(1,2,3)
    while True:
        if rl.batch_ready():
            rl.get_batch_data()

    """
    def __init__(self, seq_len, batch_size, symbol_id):
        self.symbol_id = symbol_id

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.list_of_price = []
        self.list_of_size = []
        self.list_of_time = []

    def get_id(self):
        return self.symbol_id

    def add_values(self, price, size):  # , time):
        time_ns_int = np.int64(time.time_ns())

        self.list_of_price.append(price)
        self.list_of_size.append(size)
        self.list_of_time.append(time_ns_int)

    def batch_ready(self):
        if len(self.list_of_price) >= self.seq_len + self.batch_size:
            return True

        return False

    def _get_sample(self):
        tmp_list_of_price = self.list_of_price[:self.seq_len]
        tmp_list_of_size = self.list_of_size[:self.seq_len]
        tmp_list_of_time = self.list_of_time[:self.seq_len]

        self.list_of_price.pop(0)
        self.list_of_size.pop(0)
        self.list_of_time.pop(0)

        return tmp_list_of_price, tmp_list_of_size, tmp_list_of_time

    def get_batch(self):
        x_p = []
        x_s = []
        x_t = []

        for i in range(self.batch_size):
            price_seq, size_seq, time_seq = self._get_sample()
            x_p.append(price_seq)
            x_s.append(size_seq)
            x_t.append(time_seq)

        return [np.array(x_p).astype('float32'), np.array(x_s).astype('float32'), np.array(x_t).astype(np.int64)]
