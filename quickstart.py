import imageio.v3 as iio
import torch

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

# Download the video
url = "https://github.com/facebookresearch/co-tracker/raw/main/assets/apple.mp4"
frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

device = "cuda"
grid_size = 30
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
print(video.shape)

## Run Offline CoTracker:
cotracker: CoTrackerPredictor = torch.hub.load(
    "facebookresearch/co-tracker", "cotracker2"
).to(device)
pred_tracks, pred_visibility = cotracker(
    video, grid_size=grid_size
)  # B T N 2,  B T N 1

print(pred_tracks.shape)
print(pred_visibility.shape)

## Visualize the results
vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)
