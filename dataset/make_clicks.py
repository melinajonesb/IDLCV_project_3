import numpy as np
import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
from dataloader import create_data_loaders

Coord = Tuple[int, int]  # (y, x)

class ClickSimulatorPH2:
    """
    Simulates user clicks (positive = lesion, negative = background)
    for PH2 dataset using the full segmentation masks.

    Strategies:
        - "random" : random sampling inside lesion and background (default)
        - "centroid" : centroid-based sampling (positive) + random negative
        - "boundary" : boundary-based sampling for both positive and negative (along edges, might need many clicks to work well)

    Args:
        strategy: Sampling strategy ("random", "centroid", "boundary")
        min_dist: Minimum pixel distance between clicks (optional)
        seed: Random seed (for reproducibility)
    """

    def __init__(self, strategy = "random", min_dist = 0, seed = 42):
        self.strategy = strategy.lower()
        self.min_dist = int(min_dist)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _to_numpy(self, mask: torch.Tensor) -> np.ndarray:
        """Convert tensor mask to binary numpy array."""
        mask_np = mask.detach().cpu().squeeze().numpy()
        return (mask_np > 0.5).astype(np.uint8)

    def _get_candidates(self, mask_bin: np.ndarray, positive: bool) -> np.ndarray:
        """Return indices for positive or negative regions."""
        return np.argwhere(mask_bin == (1 if positive else 0))

    def _sample_random(self, pos_coords: np.ndarray, neg_coords: np.ndarray, num_pos: int, num_neg: int):
        """
        Randomly sample coordinates for positive and negative clicks.
        Sets minimum distance if specified to avoid clustering.
        Returns: (pos_points, neg_points) as lists of (y, x)
        """
        pos = self._select_with_min_dist(pos_coords, num_pos)
        neg = self._select_with_min_dist(neg_coords, num_neg)
        return pos, neg

    def _select_with_min_dist(self, coords: np.ndarray, k: int) -> List[Coord]:
        """Select random coordinates with optional minimum spacing."""
        if len(coords) == 0:
            return []
        if self.min_dist <= 0 or len(coords) <= k:
            idx = np.random.choice(len(coords), size=min(k, len(coords)), replace=False)
            return [tuple(map(int, xy)) for xy in coords[idx]]

        chosen = []
        available = coords.copy()
        # First random
        i0 = np.random.choice(len(available))
        chosen.append(tuple(map(int, available[i0])))

        while len(chosen) < k and len(available) > 0:
            d2 = np.min([np.sum((available - np.array(c)) ** 2, axis=1) for c in chosen], axis=0)
            keep = available[d2 >= self.min_dist ** 2]
            if len(keep) == 0:
                break
            i = np.random.choice(len(keep))
            chosen.append(tuple(map(int, keep[i])))
            keep = np.delete(keep, i, axis=0)
            available = keep
        return chosen

    def _sample_centroid(self, mask_bin: np.ndarray, num_pos: int, num_neg: int):
        """
        Centroid-based clicks:
        - Positive: start at centroid (snapped to nearest positive pixel),
                    then farthest-point sampling
        - Negative: uses the _sample_random method
        Returns: (pos_points, neg_points) as lists of (y, x)
        """
        pos_coords = np.argwhere(mask_bin == 1)
        neg_coords = np.argwhere(mask_bin == 0)

        pos: List[Coord] = []
        neg: List[Coord] = []

        # --- Positive: centroid + farthest-point sampling ---
        if len(pos_coords) > 0 and num_pos > 0:
            cy, cx = np.mean(pos_coords, axis=0)            # centroid (y, x)
            c = np.array([cy, cx])
            i0 = int(np.argmin(np.sum((pos_coords - c) ** 2, axis=1)))  # nearest pos. pixel
            pos.append(tuple(map(int, pos_coords[i0])))

            remain = min(num_pos - 1, len(pos_coords) - 1)
            if remain > 0:
                min_d2 = np.sum((pos_coords - pos_coords[i0]) ** 2, axis=1)
                used = {i0}
                for _ in range(remain):
                    cand = min_d2.copy()
                    cand[list(used)] = -1
                    j = int(np.argmax(cand))
                    if cand[j] < 0:
                        break
                    pos.append(tuple(map(int, pos_coords[j])))
                    used.add(j)
                    d2_new = np.sum((pos_coords - pos_coords[j]) ** 2, axis=1)
                    min_d2 = np.minimum(min_d2, d2_new)
            pos = pos[:num_pos]

        # --- Negative: random sampling ---
        if len(neg_coords) > 0 and num_neg > 0:
            # Bruk eksisterende metode for tilfeldige negative klikk
            _, neg = self._sample_random(pos_coords, neg_coords, 0, num_neg)

        return pos, neg

    def _sample_boundary(self, mask_bin: np.ndarray, num_pos: int, num_neg: int):
        """
        Boundary-based clicks:
        - Positive clicks: along the lesion boundary
        - Negative clicks: along the outer ring just outside the lesion
        If not enough points in the ring, fall back to random background clicks.
        Returns: (pos_points, neg_points) as lists of (y, x)
        Boundary-basert klikksimulering:
        """

        mask_bin = (mask_bin > 0).astype(np.uint8)
        H, W = mask_bin.shape

        # zero padding
        pad = np.pad(mask_bin, 1, mode="constant", constant_values=0)

        # Only keep pixels that have all 8 neighbors + itself = 9 
        # (to shrink the mask so we can later extract the boundary by subtracting it)
        sum_3x3 = sum(
            pad[i:i+H, j:j+W]
            for i in range(3)
            for j in range(3)
        )
        eroded = (sum_3x3 == 9).astype(np.uint8)

        # Dilation: at least one neighbor
        dilated = (sum_3x3 > 0).astype(np.uint8)

        # Boundary og outer ring
        boundary = ((mask_bin - eroded) > 0).astype(np.uint8)
        outer_ring = ((dilated - mask_bin) > 0).astype(np.uint8)

        # --- get coordinates ---
        pos_coords = np.argwhere(boundary == 1)
        ring_coords = np.argwhere(outer_ring == 1)
        bg_coords = np.argwhere(mask_bin == 0)

        # --- choose points ---
        pos = self._select_with_min_dist(pos_coords, num_pos)

        neg = []
        if len(ring_coords) >= num_neg:
            idx = np.random.choice(len(ring_coords), size=num_neg, replace=False)
            neg = [tuple(map(int, ring_coords[i])) for i in idx]
        else:
            neg = self._select_with_min_dist(bg_coords, num_neg)

        return pos, neg

    def sample_clicks(self, mask: torch.Tensor, num_pos: int = 3, num_neg: int = 3):
        """
        Generate simulated clicks from a segmentation mask.
        Returns:
            pos_points, neg_points : lists of (y, x) coordinates
        """
        mask_bin = self._to_numpy(mask)
        pos_coords = self._get_candidates(mask_bin, positive=True)
        neg_coords = self._get_candidates(mask_bin, positive=False)

        if self.strategy == "random":
            return self._sample_random(pos_coords, neg_coords, num_pos, num_neg)
        elif self.strategy == "centroid":
            return self._sample_centroid(mask_bin, num_pos, num_neg)
        elif self.strategy == "boundary":
            return self._sample_boundary(mask_bin, num_pos, num_neg)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
    def to_click_masks(self, mask: torch.Tensor, pos_points, neg_points):
        """
        Create binary click masks [1,1,H,W] for use in loss computation.
        Returns: (pos_mask, neg_mask)
        """
        h, w = mask.shape[-2:]
        device = mask.device
        pos_mask = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
        neg_mask = torch.zeros_like(pos_mask)

        # sett punkter (y, x) til 1
        for (y, x) in pos_points:
            if 0 <= y < h and 0 <= x < w:
                pos_mask[0, 0, y, x] = 1.0
        for (y, x) in neg_points:
            if 0 <= y < h and 0 <= x < w:
                neg_mask[0, 0, y, x] = 1.0
        
        return pos_mask, neg_mask
    
    def to_loss_tensors(self, mask, pos_points, neg_points):
        pos_mask, neg_mask = self.to_click_masks(mask, pos_points, neg_points)
        click_mask = (pos_mask + neg_mask).clamp(max=1)
        click_labels = pos_mask
        
        return click_mask, click_labels
        


if __name__ == "__main__":
    print("Loading PH2 dataset...")
    train_loader, _, _ = create_data_loaders(
        dataset_name="PH2",
        batch_size=1,
        img_size=256,
        num_workers=0
    )

    # Initialiser simulatoren
    simulator = ClickSimulatorPH2(strategy="boundary", min_dist=10, seed=42)

    print("\nGenerating clicks for random masks...")
    for i, (img, mask) in enumerate(train_loader):
        if i >= 5:
            break

        pos_pts, neg_pts = simulator.sample_clicks(mask, num_pos=3, num_neg=3)
        pos_mask, neg_mask = simulator.to_click_masks(mask, pos_pts, neg_pts)

        print(f"\nImage {i+1}")
        print("  Positive clicks:", pos_pts)
        print("  Negative clicks:", neg_pts)
        print("  Positive click mask sum:", pos_mask.sum().item())
        print("  Negative click mask sum:", neg_mask.sum().item())

        # --- Visualisering ---
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normaliser for visning
        mask_np = mask[0, 0].cpu().numpy()

        vis = img_np.copy()

        def draw_dot(arr, y, x, color):
            r = 2
            y0, y1 = max(0, y - r), min(arr.shape[0], y + r + 1)
            x0, x1 = max(0, x - r), min(arr.shape[1], x + r + 1)
            arr[y0:y1, x0:x1, :] = color

        for (y, x) in pos_pts:
            draw_dot(vis, int(y), int(x), [0.0, 1.0, 0.0])  # grønne punkter = positive
        for (y, x) in neg_pts:
            draw_dot(vis, int(y), int(x), [1.0, 0.0, 0.0])  # røde punkter = negative

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(vis)
        axs[0].set_title(f"Clicks on Image {i+1}")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"click_viz_{i+1}.png")
        plt.close()
