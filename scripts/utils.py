import math
from typing import Optional

import cv2


class MP4Reader:
    def __init__(
        self,
        path: str,
        sample_intv: int = 1,
        max_out_frames: int = math.inf,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        bgr2rgb: bool = True,
    ):
        self.cap = cv2.VideoCapture(path)
        self.num_out_frames = min(
            max_out_frames, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.sample_intv = sample_intv
        self.count_frames = 0
        self.end_frame = end_frame

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.seek = start_frame
        self.bgr2rgb = bgr2rgb

    def __iter__(self):
        return self

    def __next__(self):
        if (
            self.cap.isOpened()
            and self.count_frames < self.num_out_frames
            and (self.end_frame is None or self.seek < self.end_frame)
        ):
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek)
                self.seek += self.sample_intv
                self.count_frames += 1
                if self.bgr2rgb:
                    frame = frame[..., ::-1]
                return frame
            else:
                self.cap.release()
                raise StopIteration
        else:
            raise StopIteration

    def __len__(self):
        return self.num_out_frames
