import os
from typing import Optional, Callable, Any, List, Tuple
import torchvision.datasets as datasets
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader, make_dataset


class ImageNetData(datasets.VisionDataset):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 meta_file: str = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Optional[Callable] = default_loader,
                 **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.meta_file_path = os.path.join(root, meta_file)

        super(ImageNetData, self).__init__(self.root,
                                           transform=transform,
                                           target_transform=target_transform)
        self.loader = loader
        if meta_file is None:
            classes, class_to_idx = self._find_classes(self.split_folder)
        else:
            classes, class_to_idx = self._read_meta_file()
        samples = self._make_dataset(class_to_idx)

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def _read_meta_file(self):
        instances = []
        with open(self.meta_file_path, "r") as f:
            for line in f.readlines():
                instance = [s.strip() for s in line.split() if s.strip()]
                if instance:
                    instances.append(instance)
        classes = [item[1] for item in instances]
        self.image_path = [os.path.join(self.split_folder, item[0]) for item in instances]
        self.image_target = classes
        class_sort = list(set(classes))
        class_sort.sort()
        class_to_idx = {cls: i for i, cls in enumerate(class_sort)}
        return classes, class_to_idx

    def _make_dataset(self, class_to_idx) -> List[Tuple[str, int]]:
        instances = []
        if hasattr(self, "image_path") and hasattr(self, "image_target"):
            for path, target in zip(self.image_path, self.image_target):
                instances.append((path, class_to_idx[target]))
        else:
            instances = make_dataset(self.split_folder, class_to_idx, None, None)
        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
