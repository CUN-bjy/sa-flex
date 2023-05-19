import json
import os
from typing import Optional
from typing import Callable
from typing import List
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
from slot_attention.utils import compact
from torchvision.transforms import transforms

class CLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_images = max_num_images
        self.data_path = os.path.join(data_root, "images", split)
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()

    def __getitem__(self, index: int):
        image_path = self.files[index]
        img = Image.open(image_path)
        img = img.convert("RGB")
        return self.clevr_transforms(img)

    def __len__(self):
        return len(self.files)

    def get_files(self) -> List[str]:
        with open(os.path.join(self.data_root, f"scenes/CLEVR_{self.split}_scenes.json")) as f:
            scene = json.load(f)
        paths: List[Optional[str]] = []
        total_num_images = len(scene["scenes"])
        i = 0
        while (self.max_num_images is None or len(paths) < self.max_num_images) and i < total_num_images:
            num_objects_in_scene = len(scene["scenes"][i]["objects"])
            if num_objects_in_scene <= self.max_n_objects:
                image_path = os.path.join(self.data_path, scene["scenes"][i]["image_filename"])
                assert os.path.exists(image_path), f"{image_path} does not exist"
                paths.append(image_path)
            i += 1
        return sorted(compact(paths))

class CLEVRTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
