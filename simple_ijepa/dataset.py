import torch
from torch.utils.data import Dataset
import random


class MaskedImageDataset(Dataset):

    def __init__(self, dataset, num_patches=144, num_targets=4):
        self.dataset = dataset
        self.num_patches = num_patches
        self.num_targets = num_targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        num_patches_per_dim = int(self.num_patches**0.5)
        # Generate context and target indices for this image
        context_indices, target_indices_list = self.generate_blocks(
            num_patches_per_dim)
        return img, context_indices, target_indices_list

    def sample_block(self, num_patches_per_dim, scale_range,
                     aspect_ratio_range):
        scale = random.uniform(*scale_range)
        aspect_ratio = random.uniform(*aspect_ratio_range)

        block_area = scale * num_patches_per_dim**2
        block_height = int(round((block_area * aspect_ratio)**0.5))
        block_width = int(round((block_area / aspect_ratio)**0.5))

        if block_height > num_patches_per_dim or block_width > num_patches_per_dim:
            block_height = min(block_height, num_patches_per_dim)
            block_width = min(block_width, num_patches_per_dim)

        start_row = torch.randint(0, num_patches_per_dim - block_height + 1,
                                  (1, )).item()
        start_col = torch.randint(0, num_patches_per_dim - block_width + 1,
                                  (1, )).item()

        indices = []
        for i in range(block_height):
            for j in range(block_width):
                indices.append((start_row + i) * num_patches_per_dim +
                               (start_col + j))

        return indices

    def generate_blocks(self, num_patches_per_dim):
        """Generate context and multiple target blocks"""
        context_indices = self.sample_block(num_patches_per_dim,
                                            scale_range=(0.85, 1.0),
                                            aspect_ratio_range=(0.75, 1.5))

        target_indices_list = []
        all_target_indices = set()
        for _ in range(self.num_targets):
            min_overlap = float('inf')
            best_target_indices = None
            for _ in range(
                    self.num_patches):  # Try up to num_patches iterations
                target_indices = self.sample_block(num_patches_per_dim,
                                                   scale_range=(0.15, 0.2),
                                                   aspect_ratio_range=(0.75, 1.5))
                overlap = len(set(context_indices) & set(target_indices)
                              ) + len(all_target_indices & set(target_indices))
                if overlap == 0:
                    best_target_indices = target_indices
                    break
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_target_indices = target_indices
            target_indices_list.append(best_target_indices)
            all_target_indices.update(best_target_indices)

        # Filter out overlapping indices in the context block
        context_indices = [
            idx for idx in context_indices
            if not any(idx in target_indices
                       for target_indices in target_indices_list)
        ]
        return context_indices, target_indices_list


def collate_fn(batch):
    imgs, context_indices_list, target_indices_list_list = zip(*batch)

    # Find the minimum context size and target size across the batch
    min_context_size = min(len(indices) for indices in context_indices_list)
    min_target_size = min(
        min(len(indices) for indices in target_list)
        for target_list in target_indices_list_list)

    # Truncate context and target indices to the minimum size
    context_indices_list = [
        indices[:min_context_size] for indices in context_indices_list
    ]
    target_indices_list_list = [[
        indices[:min_target_size] for indices in target_list
    ] for target_list in target_indices_list_list]

    imgs = torch.stack(imgs)
    return imgs, context_indices_list, target_indices_list_list
