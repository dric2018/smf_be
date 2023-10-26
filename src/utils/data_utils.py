"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 26 Oct, 2023
"""
import albumentations as A
import config
from matplotlib import pyplot
from tqdm import tqdm
import torch


def denormalize_img(
    normalized_image, 
    mean:list=config.MEAN, 
    std:list=config.STD
):
    denormalized_image = normalized_image.cpu().detach().clone()
    for i in range(3):
        denormalized_image[i] = (denormalized_image[i] * std[i]) + mean[i]
    
    denormalized_image = denormalized_image.permute(1, 2, 0).numpy()
    
    return denormalized_image


class History:
    def __init__(self, imgs, n_variants:int=config.NUM_HISTORY, progress_bar:bool=False):
        
        self.num_variants = n_variants
        self.progress_bar= progress_bar
        self.imgs = imgs
        self.carousel = self.generate_history(imgs)
        
    def generate_history(
        self, 
        imgs, 
        transforms:A.Compose=config.HISTORY_AUGS, 
        n_variants:int=config.NUM_HISTORY
    ):
        """
            Args:
                TBA

            Returns:
                augmented_images: Tensor of shape (B, C, num_variants, H, W) containing the variants of the perceived images
        """
        B, C, H, W = imgs.shape

        augmented_images = torch.empty((B, C, self.num_variants+1, H, W))
        if self.progress_bar:
            pbar = tqdm(range(imgs.shape[0]), desc="Generating history")
        else:
            pbar = range(imgs.shape[0])
        for i in pbar:
            # Get the original image
            original_image = imgs[i]
            augmented_images[i, :, 0] = original_image # insert original image first
            
            # Apply augmentations and store in the final tensor
            for j in range(1, self.num_variants):
                augmented_image = transforms(image=original_image.permute(1, 2, 0).cpu().detach().numpy())['image']
                augmented_images[i, :, j] = torch.from_numpy(augmented_image).permute(2, 0, 1)

        return augmented_images


    def display_history(self):
        
        if self.carousel is None:
            self.generate_history(self.imgs)

        B, C, num_variants, H, W = self.carousel.shape

        fig, axes = pyplot.subplots(B, num_variants, figsize=(12, 16))

        for i in range(B):
            # Display the original image
            original_image = denormalize_img(self.imgs[i])
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Original {i + 1}")
            axes[i, 0].axis('off')

            # Display the 5 augmented variants
            for j in range(5):
                variant_image = denormalize_img(self.carousel[i, :, j])
                axes[i, j + 1].imshow(variant_image)
                axes[i, j + 1].set_title(f"Variant {j + 1}")
                axes[i, j + 1].axis('off')

        pyplot.tight_layout()

        pyplot.show()

