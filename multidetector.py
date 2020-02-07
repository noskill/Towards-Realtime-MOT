from tracker.multitracker import JDETracker, STrack, mot_detector
from substractor import SubstracktorDetector


class SubstractorTracker(JDETracker):
    def __init__(self, opt, frame_rate=30):
        JDETracker.__init__(self, opt, frame_rate=frame_rate)
        self.substractor = SubstracktorDetector()


    def update(self, img0, im_blob):
        detections = mot_detector(img0, im_blob, self.model,
                                  self.opt.conf_thres,
                                  self.opt.nms_thres,
                                  self.opt.img_size)
        self.frame_id += 1
        # output_stracks = self._update(detections)
        detections1 = self.substractor.detect(img0)
        output_stracks = self._update(detections1)
        return output_stracks
