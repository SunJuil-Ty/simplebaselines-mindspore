"""
preprocess.
"""
import os
import numpy as np

from src.dataset import keypoint_dataset
from src.config import config

def get_bin():
    ''' get bin files'''
    valid_dataset, _ = keypoint_dataset(
        config,
        bbox_file=config.TEST.COCO_BBOX_FILE,
        train_mode=False,
        num_parallel_workers=config.TEST.NUM_PARALLEL_WORKERS,
    )
    inputs_path = os.path.join(config.INFER.PRE_RESULT_PATH, "00_data")
    os.makedirs(inputs_path)

    center_path = os.path.join(config.INFER.PRE_RESULT_PATH, "center")
    os.makedirs(center_path)

    scale_path = os.path.join(config.INFER.PRE_RESULT_PATH, "scale")
    os.makedirs(scale_path)

    score_path = os.path.join(config.INFER.PRE_RESULT_PATH, "score")
    os.makedirs(score_path)

    id_path = os.path.join(config.INFER.PRE_RESULT_PATH, "id")
    os.makedirs(id_path)

    for i, item in enumerate(valid_dataset.create_dict_iterator(output_numpy=True)):
        file_name = "sp_bs" + str(config.TEST.BATCH_SIZE) + "_" + str(i) + ".bin"
        # input data
        inputs = item['image']
        inputs_file_path = os.path.join(inputs_path, file_name)
        inputs.tofile(inputs_file_path)
        if config.TEST.FLIP_TEST:
            inputs_flipped = inputs[:, :, :, ::-1]
            file_name = "sp_flip_bs" + str(config.TEST.BATCH_SIZE) + "_" + str(i) + ".bin"
            inputs_file_path = os.path.join(inputs_path, file_name)
            inputs_flipped.tofile(inputs_file_path)
        file_name = "sp_bs" + str(config.TEST.BATCH_SIZE) + "_" + str(i) + ".npy"
        np.save(os.path.join(center_path, file_name), item['center'])
        np.save(os.path.join(scale_path, file_name), item['scale'])
        np.save(os.path.join(score_path, file_name), item['score'])
        np.save(os.path.join(id_path, file_name), item['id'])
    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    get_bin()
