# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
import glob
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import csv
import zipfile
import cv2
import imageio.v2 as imageio

from pointcept.datasets.preprocessing.scannet.SensorData import SensorData


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    mapping = dict()
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k, v in label_mapping.items():
        mapped[image == k] = v
    return mapped.astype(np.uint8)
    

def handle_process(
    scene_path, output_path, limit, label_map, parse_depth=True, parse_color=True, 
    parse_poses=True, parse_intrinsics=True, parse_label=True
):
    scene_id = os.path.basename(scene_path)
    sens_path = os.path.join(scene_path, scene_id + ".sens")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, scene_id))

    # load the data
    print(f"Processing {scene_id} of {sens_path}")
    sd = SensorData(sens_path, limit=limit)
    if parse_depth:
        image_size = sd.export_depth_images(os.path.join(output_path, scene_id, "depth"))
        H, W = image_size
        assert H == 480 and W == 640, f"Depth image size is {H}x{W}, not 480x640"
    if parse_color:
        sd.export_color_images(os.path.join(output_path, scene_id, "color"), image_size=image_size)
    if parse_poses:
        sd.export_poses(os.path.join(output_path, scene_id, "pose"))
    if parse_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, scene_id, "intrinsic"))

    os.system(f"cp {scene_path}/scene*.txt {output_path}/{scene_id}/")

    if parse_label:
        label_zip_path = os.path.join(
            scene_path, f"{scene_id}_2d-label-filt.zip"
        )
        with open(label_zip_path, "rb") as f:
            zip_file = zipfile.ZipFile(f)
            for frame in range(0, len(sd.frames)):
                label_file = f"label-filt/{frame}.png"
                with zip_file.open(label_file) as lf:
                    image = np.array(imageio.imread(lf))

                mapped_image = map_label_image(image, label_map)
                output_root = os.path.join(output_path, scene_id, "label")
                os.makedirs(output_root, exist_ok=True)
                cv2.imwrite(os.path.join(output_root, f"{frame}.png"), mapped_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", 
        required=True, 
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root", 
        required=True, 
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--images_per_scene", type=int, default=300, help="Number of images to export per scene"
    )
    parser.add_argument(
        "--export_depth", action="store_true", default=False, help="Whether to export depth images"
    )
    parser.add_argument(
        "--export_color", action="store_true", default=False, help="Whether to export color images"
    )
    parser.add_argument(
        "--export_pose", action="store_true", default=False, help="Whether to export poses"
    )
    parser.add_argument(
        "--export_intrinsic", action="store_true", default=False, help="Whether to export intrinsics"
    )
    parser.add_argument(
        "--export_label", action="store_true", default=False, help="Whether to export labels"
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()
    
    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/scans/scene*"))
    scene_paths += sorted(glob.glob(config.dataset_root + "/scans_test/scene*"))

    label_mapping = None
    if config.export_label:
        root = os.path.dirname(config.dataset_root)
        label_map = read_label_mapping(
            filename="pointcept/datasets/preprocessing/scannet/meta_data/scannetv2-labels.combined.tsv",
            label_from="id",
            label_to="nyu40id",
        )
    else:
        label_map = None

    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(config.images_per_scene),
            repeat(label_map),
            repeat(config.export_depth),
            repeat(config.export_color),
            repeat(config.export_pose),
            repeat(config.export_intrinsic),
            repeat(config.export_label),
        )
    )

    