from utils import my_sql_connector as sql
import uuid
import logging
import numpy as np
import const.data_inf_prod_consts as DC
from lib.numtry import NumTry
from itertools import islice
from numpy import newaxis

logging.basicConfig(filename="res/inference_results.log", format='%(asctime)s %(message)s', filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class InferenceManager:
    def __init__(self, bm_):
        self.nt = NumTry()
        self.bm = bm_

    def perform_inference(self, model, model_update_counter, model_name_previous=None):
        if model_name_previous is None:  # during training its None
            model_name = str(model_update_counter) + '__' + str(uuid.uuid4())
            model_name_to_save = 'res/' + model_name + '.h5'
            # model.save(model_name_to_save)
            model.save_weights(model_name_to_save)
        else:  # during inference it will be a string
            model_name = model_name_previous.replace(".h5", "")

        if not DC.PERFORM_INFERENCE:
            return 0

        inference_table_names = self.bm.get_inference_table_names()
        inf_result_manager = InferenceResultsManager(model_name)

        for table_name_spec in inference_table_names:
            table_name = table_name_spec.get_table_name()
            all_table_rows = sql.get_all_rows_from_table(table_name)
            print('performing model inference on ', table_name)

            signal_points, present_points = self.infer_window_by_window(all_table_rows, DC.PAST_LENGTH, model)

            fn0 = 'res/' + table_name + '__' + model_name + '.txt'
            column_stack0 = np.column_stack((present_points, signal_points))
            np.savetxt(fn0, column_stack0, fmt='%s')

            if not DC.PERFORM_BACKTEST:
                continue

            present_points = self.nt.simple_moving_average_1d(present_points, DC.BACKTEST_PRICE_SMA_PERIOD)
            for stop_loss in DC.BACKTEST_STOP_LOSS_VALUES_USD:
                for target in DC.BACKTEST_TARGET_VALUES_USD:
                    for buy_thrsh in DC.BACKTEST_BUY_SIGNAL_THRESHOLDS:
                        for sel_thrsh in DC.BACKTEST_SELL_SIGNAL_THRESHOLDS:
                            eq_lst, long_tr_lst, short_tr_lst, pl, max_eq, min_eq, num_win_tr, num_los_tr, num_long_tr, num_short_tr = self.nt.backtest(
                                signal_points, DC.SIGNAL_BUY_INDEX, DC.SIGNAL_SELL_INDEX, present_points, buy_thrsh, sel_thrsh, target, stop_loss)

                            backtest_str = 'pl {}, max_eq {}, min_eq {}, n_win_tr {}, n_los_tr {}, n_long_tr {}, n_short_tr {}, buy_thr {}, sel_thr {}, targ {}, stop {}'.format(
                                pl, max_eq, min_eq, num_win_tr, num_los_tr, num_long_tr, num_short_tr, buy_thrsh, sel_thrsh, target, stop_loss)
                            print(backtest_str)

                            if DC.SAVE_INFERENCE_FILES and pl > DC.MIN_PL_TO_SAVE_INFERENCE_FILES:
                                fn_1 = '{}_tab_nam_{}_pl_{}_max_eq_{}_min_eq_{}_n_win_tr_{}_n_los_tr_{}_n_long_tr_{}_n_short_tr_{}_buy_thr_{}_sel_thr_{}_targ_{}_stop_{}_'.format(
                                    model_name, table_name, pl, max_eq, min_eq, num_win_tr, num_los_tr, num_long_tr, num_short_tr, buy_thrsh, sel_thrsh, target, stop_loss)
                                fn_2 = 'res/' + fn_1 + '.txt'
                                column_stack2 = np.column_stack((present_points, signal_points, long_tr_lst, short_tr_lst, eq_lst))
                                np.savetxt(fn_2, column_stack2, fmt='%s')

                            inf_result_manager.add_backtest_results(table_name, buy_thrsh, sel_thrsh, target, stop_loss, pl, num_win_tr, num_los_tr, max_eq, min_eq, num_long_tr, num_short_tr)

        inf_result_manager.create_report()

    def infer_window_by_window(self, all_table_rows, past_len, model):
        signal_out = []
        price_out = []

        table_size_min_past = len(all_table_rows) - DC.PAST_LENGTH
        counter = 0

        it = iter(all_table_rows)
        result = tuple(islice(it, past_len))
        if len(result) == past_len:
            signal, price_point = self.infer_window_by_window_helper(result, model)
            signal_out.append(signal)
            price_out.append(price_point)

            print('rows remaining', table_size_min_past - counter)
            counter = counter + 1

        for elem in it:
            result = result[1:] + (elem,)
            signal, price_point = self.infer_window_by_window_helper(result, model)
            signal_out.append(signal)
            price_out.append(price_point)

            print('rows remaining', table_size_min_past - counter)
            counter = counter + 1

        return signal_out, price_out

    def infer_window_by_window_helper(self, window, model):
        price_window, size_window, time_window = self.nt.separate_price_size_time_ib_1d(window, DC.IB_PRICE_INDEX, DC.IB_SIZE_INDEX, DC.IB_TIME_INDEX)

        price_window_batch = np.array(price_window)[newaxis, :]
        size_window_batch = np.array(size_window)[newaxis, :]
        time_window_batch = np.array(time_window)[newaxis, :]

        signal = model([price_window_batch, size_window_batch, time_window_batch], training=False)[0]

        return signal, price_window[-1]


class InferenceResultsManager:
    def __init__(self, model_name_):
        self.model_name = model_name_
        self.test_runs_list = []

    def add_backtest_results(self, table_name__, buy_thr_, sel_thr_, target_, stop_loss_, pl, num_win_tr, num_los_tr, max_eq, min_eq, num_long_tr, num_short_tr):

        parameter_string__ = 'buy_thr,' + str(buy_thr_) + ',sel_thr,' + str(sel_thr_) + ',target,' + str(target_) + ',stop_loss,' + str(stop_loss_)

        test_run = TestRun(table_name__, parameter_string__, buy_thr_, sel_thr_, target_, stop_loss_, pl, num_win_tr, num_los_tr, max_eq, min_eq, num_long_tr, num_short_tr)
        self.test_runs_list.append(test_run)

    def create_report(self):
        all_param_strings = []
        for test_run in self.test_runs_list:
            all_param_strings.append(test_run.parameter_string)

        unique_param_strings = np.unique(np.array(all_param_strings))

        for unique_string in unique_param_strings:
            tot_pl_s = 'pl_list'

            tot_win_trds_s = 'win_trds_list'
            tot_los_trds_s = 'los_trds_list'

            max_eq_s = 'max_eq_list'
            min_eq_s = 'min_eq_list'

            num_long_tr_s = 'long_tr_list'
            num_short_tr_s = 'short_tr_list'

            for test_run in self.test_runs_list:
                if str(test_run.parameter_string) == str(unique_string):
                    tot_pl_s = tot_pl_s + ',' + str(test_run.pl)

                    tot_win_trds_s = tot_win_trds_s + ',' + str(test_run.num_win_tr)
                    tot_los_trds_s = tot_los_trds_s + ',' + str(test_run.num_los_tr)

                    max_eq_s = max_eq_s + ',' + str(test_run.max_eq)
                    min_eq_s = min_eq_s + ',' + str(test_run.min_eq)

                    num_long_tr_s = num_long_tr_s + ',' + str(test_run.num_long_tr)
                    num_short_tr_s = num_short_tr_s + ',' + str(test_run.num_short_tr)

            to_log1 = 'mod,' + self.model_name + ',' + unique_string + ',' + tot_win_trds_s + ',' + tot_los_trds_s
            to_log2 = tot_pl_s + ',' + max_eq_s + ',' + min_eq_s + ',' + num_long_tr_s + ',' + num_short_tr_s
            to_log3 = to_log1 + ',' + to_log2
            logger.info(to_log3)

        return True


class TestRun:
    def __init__(self, table_name__, parameter_string__, buy_thr_, sel_thr_, target_, stop_loss_, pl, num_win_tr, num_los_tr, max_eq, min_eq, num_long_tr, num_short_tr):
        self.table_name = table_name__
        self.parameter_string = parameter_string__

        self.buy_thrsh = buy_thr_
        self.sel_thrsh = sel_thr_

        self.target = target_
        self.stop_loss = stop_loss_

        self.pl = pl
        self.num_win_tr = num_win_tr
        self.num_los_tr = num_los_tr

        self.max_eq = max_eq
        self.min_eq = min_eq

        self.num_long_tr = num_long_tr
        self.num_short_tr = num_short_tr

