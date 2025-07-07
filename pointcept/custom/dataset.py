import os
import numpy as np

from pointcept.utils.cache import shared_dict
from pointcept.datasets.defaults import DefaultDataset
from pointcept.datasets.builder import DATASETS

from .custom_utils.io import load_bytes, imfrombytes


# Custom Dataset
@DATASETS.register_module()
class ScanNetRGBDDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
    ]
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        rgb_root="data/scannet/rgbd",
        num_cameras=5,
        **kwargs,
    ):
        self.num_cameras = num_cameras
        self.rgbd_root = rgb_root

        super().__init__(split=split, data_root=data_root, **kwargs)

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = os.path.basename(data_path)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)
        
        split = data_path.split("/")[-2]
        metadata = np.load(
            os.path.join(self.data_root, "metadata", split, name + ".npy"),
            allow_pickle=True
        ).item()
        assert name == metadata["scene_name"]
        intrinsic = metadata["intrinsic"]
        frames = metadata["frames"]
        
        frame_idxs = np.random.choice(
            list(frames.keys()), self.num_cameras, replace=self.num_cameras > len(frames)
        )
        frames = [frames[frame_idx] for frame_idx in frame_idxs]
        intrinsics = np.array(intrinsic)
        extrinsics = np.stack([frame["extrinsic"] for frame in frames], axis=0)
        
        rgb, depth = [], []
        for frame in frames:
            depth_bytes = load_bytes(os.path.join(self.rgbd_root, frame["depth_path"]))
            rgb_bytes = load_bytes(os.path.join(self.rgbd_root, frame["color_path"]))
            depth_im = imfrombytes(depth_bytes, flag="unchanged").astype(np.float32) # H, W
            rgb_im = imfrombytes(rgb_bytes, flag="color", channel_order="rgb").astype(np.float32) # H, W, 3
            rgb.append(rgb_im)
            depth.append(depth_im)
        data_dict = {}
        data_dict["intrinsic"] = intrinsics.astype(np.float32)
        data_dict["extrinsic"] = extrinsics.astype(np.float32)
        data_dict["rgb"] = np.stack(rgb, axis=0)
        data_dict["depth"] = np.stack(depth, axis=0)
        data_dict["depth_scale"] = 1.0 / 1000.0

        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        return data_dict

