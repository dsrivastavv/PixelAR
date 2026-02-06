from pathlib import Path
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir, preprocessed_entropy_dir=None):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.preprocessed_entropy_dir = preprocessed_entropy_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files = sorted(os.listdir(feature_dir))
        self.label_files = sorted(os.listdir(label_dir))
        
        # filter only .npy files
        self.feature_files = [f for f in self.feature_files if f.endswith('.npy')]
        self.label_files = [f for f in self.label_files if f.endswith('.npy')]
        
        # sort by index
        self.feature_files.sort(key=lambda x: int(x.split('.')[0]))
        self.label_files.sort(key=lambda x: int(x.split('.')[0]))
        
        # TODO: make it configurable
        # self.feature_files = [f"{i}.npy" for i in range(1281167)]
        # self.label_files = [f"{i}.npy" for i in range(1281167)]
        # NOTE: Wierd logic copied from original codebase
        self.feature_files = self.feature_files[:1281167]
        self.label_files = self.label_files[:1281167]
        
        print(f"Found {len(self.feature_files)} feature files and {len(self.label_files)} label files in {feature_dir} and {label_dir} respectively.")

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))

        entropies = None
        if self.preprocessed_entropy_dir:
            entropy_path = Path(self.preprocessed_entropy_dir, *Path(feature_dir).parts[-3:], feature_file.replace(".npy", ".pt"))
            entropies = torch.load(entropy_path)
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
            entropies = entropies[aug_idx] if entropies is not None else None
        labels = np.load(os.path.join(label_dir, label_file))
        if self.preprocessed_entropy_dir:
            return torch.from_numpy(features[0]), labels[0], entropies
        else:
            return torch.from_numpy(features[0]), labels[0]


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)