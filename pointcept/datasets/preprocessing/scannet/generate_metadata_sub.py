import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
import glob
import numpy as np
from collections import defaultdict
from multiprocessing import Pool


nearby_num = 2
nearby_interval = 20
frame_interval = 10
META_NAME = "metadata"


def save_meta(config, scene_name, split):
    frame_list = os.listdir(os.path.join(config.rgbd_root, scene_name, "color"))
    frame_list = list(frame_list)
    frame_list = [frame for frame in frame_list if frame.endswith(".jpg")]
    frame_list.sort(key=lambda x: int(x.split(".")[0]))
    frame_list = frame_list[::10]

    intrinsic = np.loadtxt(
        os.path.join(config.rgbd_root, scene_name, "intrinsic", "intrinsic_depth.txt")
    )
    if intrinsic is None or np.isnan(intrinsic).any() or np.isinf(intrinsic).any():
        return
    meta_data = defaultdict(dict)
    meta_data["scene_name"] = scene_name
    meta_data["intrinsic"] = intrinsic
    meta_data["frames"] = defaultdict(dict)
    for data in frame_list:
        pose = np.loadtxt(
            os.path.join(config.rgbd_root, scene_name, "pose", data.replace(".jpg", ".txt"))
        )
        if pose is None or np.isnan(pose).any() or np.isinf(pose).any():
            continue
        # from c2w to w2c
        pose = np.linalg.inv(pose)
        if pose is None or np.isnan(pose).any() or np.isinf(pose).any():
            continue

        color_path = os.path.join(scene_name, "color", data)
        meta_data["frames"][data]["color_path"] = color_path

        depth_path = os.path.join(scene_name, "depth", data.replace(".jpg", ".png"))
        meta_data["frames"][data]["depth_path"] = depth_path
        meta_data["frames"][data]["extrinsic"] = pose
    save_path = os.path.join(config.dataset_root, META_NAME, split, f"{scene_name}.npy")
    np.save(save_path, meta_data)
    print(f"Saved {scene_name} metadata to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", 
        default="data/scannet",
        required=True, 
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--rgbd_root", 
        default="data/scannet/rgbd",
        required=True, 
        help="Path to the ScanNet dataset containing rgbd folders",
    )
    config = parser.parse_args()
    
    for split in ["train", "val", "test"]:
        scene_list = glob.glob(os.path.join(config.dataset_root, split, "*"))
        output_dir = os.path.join(config.dataset_root, META_NAME, split)
        os.makedirs(output_dir, exist_ok=True)

        with Pool(processes=8) as pool:
            for scene in scene_list:
                scene_name = os.path.basename(scene)
                pool.apply_async(save_meta, args=(config, scene_name, split))

            pool.close()
            pool.join()
