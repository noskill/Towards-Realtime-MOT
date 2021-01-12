import numpy
import cv2
import message_filters
import rospy
from tracker.ext_tracker import Detection, ExternalDetTracker
from jpda_rospack.msg import detection3d_with_feature_array, \
    detection3d_with_feature, detection2d_with_feature_array
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from utils import visualization as vis


bridge = CvBridge()


class RosTracker(ExternalDetTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tracks = [] 

    def det2d_callback(self, msg):
        new_det = []
        for det in msg.detection2d_with_features:
            x1, x2, y1, y2 = det.x1, det.x2, det.y1, det.y2
            top_left = [min(x1, x2), min(y1, y2)]
            w = abs(x1 - x2)
            h = abs(y1 - y2)
            feature = numpy.asarray(det.feature)
            conf = det.confidence
            tlwh = numpy.asarray(top_left + [w, h])
            new_det.append(Detection(tlwh, feature, confidence=conf))
        tracks = self.update(new_det) 
        self.last_tracks = tracks
        return tracks
    
    def callback_img(self, msg_img):
        cv2_img = bridge.imgmsg_to_cv2(msg_img, "bgr8")
        self.process_img(cv2_img)

    def process_img(self, cv2_img):
        online_targets = self.last_tracks 
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 100 and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        online_im = vis.plot_tracking(cv2_img, online_tlwhs, online_ids, frame_id=self.frame_id,
                                          fps=20)
        cv2.imshow('tracking', online_im)
        cv2.waitKey(200)


def main():
    rospy.init_node("RealtimeMot")
    draw_img = True
    tracker = RosTracker(buffer_size=90)
    _2d_bbox_features_topic = rospy.get_param("2d_bbox_features_topic")
    if draw_img:
        raw_image_topic = rospy.get_param('raw_image_topic')
        rospy.Subscriber(raw_image_topic, Image, 
                tracker.callback_img, queue_size=5)
    rospy.Subscriber(_2d_bbox_features_topic,
                 detection2d_with_feature_array, tracker.det2d_callback)
    
    rospy.spin()

if __name__ == '__main__':
    main()
