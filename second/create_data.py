
import fire
import copy
from pathlib import Path
import pickle
import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database

# Kitti dataset preparation
def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    kitti_dataset_name = "KittiDataset"
    kitti_pkl_name = "kitti_infos_train.pkl"
    create_groundtruth_database(kitti_dataset_name, root_path, Path(root_path) / kitti_pkl_name)

# Nuscenes data preparation
def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    pkl_name = "infos_train.pkl"
    if version == "v1.0-test": name = "infos_test.pkl"
    create_groundtruth_database(dataset_name, root_path, Path(root_path) / pkl_name)


if __name__ == '__main__':
    fire.Fire()
