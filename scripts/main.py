import cv2
import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
from scripts.utils import MP4Reader


def main():
    device = "cuda"

    # Read mask and video
    mask_path = "/home/feishi/cod/epic-kitchen_process/outputs/masks/P01_01_interpolations_right hand_masks.mp4"
    mask = next(MP4Reader(mask_path))
    print(f"{mask.shape=}")

    video_path = "/home/feishi/data/EPIC-KITCHENS/P01/videos/P01_01.MP4"
    reader = MP4Reader(video_path, start_frame=936, max_out_frames=50)
    frames = np.array([next(reader) for _ in range(len(reader))])
    cv2.imwrite("frame.png", frames[0])
    print(f"{frames.shape=}")

    # Resize frames to match the mask dimensions using PyTorch
    mask_h, mask_w = mask.shape[:2]
    frames = (
        torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    )  # Convert to (B, C, H, W)
    frames = torch.nn.functional.interpolate(
        frames,
        size=(mask_h, mask_w),
        mode="bilinear",
        align_corners=False,
    )
    frames = frames[None].to(device)  # B T C H W
    print(f"{frames.shape=}")

    mask = mask[..., 0][None, None]  # B 1 H W
    mask = torch.from_numpy(mask).to(device)

    # Run CoTracker
    grid_size = 30
    cotracker: CoTrackerPredictor = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker2"
    ).to(device)

    pred_tracks, pred_visibility = cotracker(
        frames, segm_mask=mask, grid_size=grid_size
    )  # B T N 2,  B T N 1

    # Visualize
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(frames, pred_tracks, pred_visibility)


if __name__ == "__main__":
    main()
