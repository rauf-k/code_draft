import numpy as np
from utils import my_sql_connector as sql
from const import data_inf_prod_consts as DC


class BatchManager:
    def __init__(self):
        self.list_of_suitable_TableObjects = self.get_list_of_suitable_TableSpecs()
        self.inference_tables = self.list_of_suitable_TableObjects[-DC.NUMBER_OF_TABLES_TO_RESERVE_FOR_INFERENCE:]
        self.training_tables = self.list_of_suitable_TableObjects[:-DC.NUMBER_OF_TABLES_TO_RESERVE_FOR_INFERENCE]

    def get_inference_table_names(self):
        return self.inference_tables

    def get_training_table_names(self):
        return self.training_tables

    def get_list_of_suitable_TableSpecs(self):
        print('getting a list of suitable tables for training and inference...')
        oot = []
        index = 0
        list_of_all_table_names = sql.get_table_names(DC.SYMBOL)
        for table_name in list_of_all_table_names:
            num_of_rows = sql.get_number_of_rows_in_a_table(table_name)
            if num_of_rows > DC.PAST_LENGTH + DC.FUTURE_LENGTH + 100:
                ts = TableObject(index, table_name, num_of_rows)
                print(index, table_name, num_of_rows)
                oot.append(ts)
                index = index + 1

        print('')
        return oot

    def get_batch_descriptors_helper(self, number_of_rows_in_table, number_of_samples_in_batch, past_len_, future_len_):
        table_sections_oot = []

        sequence_len = past_len_ + future_len_
        table_seg_len = sequence_len + number_of_samples_in_batch
        start_index = 0
        end_index = table_seg_len

        for table_row in range(number_of_rows_in_table):
            table_sections_oot.append((start_index, end_index))

            start_index = end_index - sequence_len + 1
            end_index = start_index + table_seg_len

            if end_index > number_of_rows_in_table:
                break

        return table_sections_oot


    def get_raw_batch_1d(self, batch_descriptor):
        arr1d = sql.get_specific_rows_from_table(batch_descriptor.get_table_name(), batch_descriptor.get_start_row(), batch_descriptor.get_end_row())
        np_arr = np.array(arr1d)

        return np_arr


class BatchObject:
    def __init__(self, table_name_, start_row_index_, end_row_index_, index_):
        self.table_name = table_name_
        self.start_row_index = start_row_index_
        self.end_row_index = end_row_index_
        self.index = index_

    def get_table_name(self):
        return self.table_name

    def get_start_row(self):
        return self.start_row_index

    def get_end_row(self):
        return self.end_row_index

    def get_index(self):
        return self.index


class TableObject:
    def __init__(self, index_, table_name_, number_of_rows_):
        self.index = index_
        self.table_name = table_name_
        self.number_of_rows = number_of_rows_

    def get_index(self):
        return self.index

    def get_table_name(self):
        return self.table_name

    def get_number_of_rows(self):
        return self.number_of_rows
