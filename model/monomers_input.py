import numpy as np

def perform_augmentation(number_table, sub_max_len, sub_pad_id):
    """augmentation"""
    aug_table = []
    aug_nums = []
    aug_ID = []

    for i in range(len(number_table)):
        num_now = len(np.where(number_table[i] !=  sub_pad_id)[0])
        table_now = number_table[i][np.where(number_table[i] != sub_pad_id)]

        # rotation
        for j in range(num_now):
            # rotate number
            table_now = list(table_now[1:]) + [table_now[0]]
            # translation
            for k in range((sub_max_len-num_now)+1):
                pad_start = [sub_pad_id for l in range(k)]
                pad_end = [sub_pad_id for l in range(sub_max_len-num_now-k)]

                tmp_ = table_now
                tmp_ = pad_start + tmp_
                tmp_ = tmp_ + pad_end

                aug_table.append(tmp_)
                aug_nums.append(num_now)
                aug_ID.append(i+1)
    # list
    return aug_table, aug_nums, aug_ID


def generate_feature_map(number_table, data_preprocessing, sub_pad_val, sub_pad_id):
    """generate feature_map from number_table & data_preprocessing"""
    pad = [sub_pad_val] * data_preprocessing.shape[1]
    feature_map = []
    for i in range(len(number_table)):
        feature_map_now = [pad if _ == sub_pad_id else data_preprocessing[_-1].tolist() for _ in number_table[i]]
        feature_map.append(feature_map_now)
    feature_map = np.array(feature_map).transpose(0,2,1)

    return feature_map