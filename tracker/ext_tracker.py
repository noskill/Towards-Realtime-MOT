from .multitracker import JDETracker, STrack
from utils.kalman_filter import KalmanFilter


class Detection:
    def __init__(self, tlwh, features, confidence=0.6):
        """
        Parameters
        ----------
        tlwh: tuple[int]
            top-left, width, height format
        features: numpy.array
            reid features
        confidence: float
            detector confidence score
        """
        self.tlwh = tlwh
        self.features = features
        self.confidence = confidence


class ExternalDetTracker(JDETracker):
    def __init__(self, buffer_size=30, det_threshold=0.4):
        self.det_thresh = det_threshold
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.buffer_size = buffer_size 
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def update(self, detections, img=None):
        """
        Process detections

        Parameters
        ----------
        detections: List[Detection]
        """
        stracks = [STrack(det.tlwh, 
            det.confidence, det.features, 30) for det in detections]
        return self._update(stracks)

