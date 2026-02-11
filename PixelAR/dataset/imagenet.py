import torch
import random
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PixelAR.utils.image import center_crop_arr
from torchvision import transforms
from typing import List
from PIL import Image

class ImageNetTenCropDataset(Dataset):
    def __init__(self, root_dir, image_size: int = 256, crop_ranges: List[float] = [1.05, 1.1], ten_crop: bool = True, patch_size: int = 16):
        self.dataset = ImageFolder(root_dir)
        self.patch_size = patch_size
        
        if ten_crop:
            self.transforms = []
            for crop_range in crop_ranges:
                crop_size = int(image_size * crop_range) 
                transform = transforms.Compose([
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                    transforms.TenCrop(image_size), # this is a tuple of PIL Images
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                self.transforms.append(transform)
        else:
            crop_size = image_size 
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            self.transforms = [transform]
        
            
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        random_transform = random.choice(self.transforms)
        image = random_transform(image)
        if len(image.shape) == 4:  # If TenCrop is applied, we have a batch of crops
            # randomly select one of the crops
            crop_idx = random.randint(0, image.shape[0] - 1)
            image = image[crop_idx]
            
        # map image back to unint8 values
        image = (image * 255).to(torch.uint8)    # [3, H, W]
        
        # convert to [T, 3, patch_size, patch_size] in a raster scan order
        H, W = image.shape[1], image.shape[2]
        H_, W_ = H // self.patch_size, W // self.patch_size
        image = image.reshape(3, H_, self.patch_size, W_, self.patch_size)
        image = torch.einsum('chpwq->hwcpq', image)
        image = image.reshape(H_ * W_, 3, self.patch_size, self.patch_size).contiguous() # [T, 3, patch_size, patch_size]        
        # image = torch.randint(0, 256, (256,), dtype=torch.uint8)
        return image, label
    
    def visualize(self, idx):
        image, _ = self.__getitem__(idx) # (T, 3, patch_size, patch_size)
        
        # visualize each patch and stitch them together
        patch_images = []
        for patch_img in image:
            patch_img = patch_img.permute(1, 2, 0).numpy().astype("uint8") # (patch_size, patch_size, 3)
            patch_images.append(Image.fromarray(patch_img))
        
        # stitch patches together into a 2D grid with space between them
        H_ = W_ = int((image.shape[0]) ** 0.5)
        grid_image = Image.new('RGB', (W_ * self.patch_size + (W_ - 1) * 1, H_ * self.patch_size + (H_ - 1) * 1), (255, 255, 255))
        for i in range(H_):
            for j in range(W_):
                patch_img = patch_images[i * W_ + j]
                grid_image.paste(patch_img, (j * (self.patch_size + 1), i * (self.patch_size + 1)))

        return grid_image
        
        
    def __len__(self):
        return len(self.dataset)

# class CustomDataset(Dataset):
#     def __init__(self, feature_dir, label_dir):
#         self.feature_dir = feature_dir
#         self.label_dir = label_dir
#         self.flip = 'flip' in self.feature_dir

#         aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
#         aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
#         if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
#             self.aug_feature_dir = aug_feature_dir
#             self.aug_label_dir = aug_label_dir
#         else:
#             self.aug_feature_dir = None
#             self.aug_label_dir = None

#         self.feature_files = sorted(os.listdir(feature_dir))
#         self.label_files = sorted(os.listdir(label_dir))
        
#         # filter only .npy files
#         self.feature_files = [f for f in self.feature_files if f.endswith('.npy')]
#         self.label_files = [f for f in self.label_files if f.endswith('.npy')]
        
#         # sort by index
#         self.feature_files.sort(key=lambda x: int(x.split('.')[0]))
#         self.label_files.sort(key=lambda x: int(x.split('.')[0]))
        
#         # TODO: make it configurable
#         # self.feature_files = [f"{i}.npy" for i in range(1281167)]
#         # self.label_files = [f"{i}.npy" for i in range(1281167)]
#         # NOTE: Wierd logic copied from original codebase
#         self.feature_files = self.feature_files[:1281167]
#         self.label_files = self.label_files[:1281167]
        
#         print(f"Found {len(self.feature_files)} feature files and {len(self.label_files)} label files in {feature_dir} and {label_dir} respectively.")

#     def __len__(self):
#         assert len(self.feature_files) == len(self.label_files), \
#             "Number of feature files and label files should be same"
#         return len(self.feature_files)

#     def __getitem__(self, idx):
#         if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
#             feature_dir = self.aug_feature_dir
#             label_dir = self.aug_label_dir
#         else:
#             feature_dir = self.feature_dir
#             label_dir = self.label_dir
                   
#         feature_file = self.feature_files[idx]
#         label_file = self.label_files[idx]

#         features = np.load(os.path.join(feature_dir, feature_file))

#         if self.flip:
#             aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
#             features = features[:, aug_idx]
#         labels = np.load(os.path.join(label_dir, label_file))
#         return torch.from_numpy(features[0]), labels[0]
        
#     def visualize(self, idx):
#         image, _ = self.__getitem__(idx) # (3, H, W)
#         image = image.permute(1, 2, 0).numpy().astype("uint8") # (H, W, 3)
        
#         # convert to PIL Image for visualization
#         image = Image.fromarray(image)
#         return image