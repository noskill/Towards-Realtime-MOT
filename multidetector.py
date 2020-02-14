from tracker.multitracker import JDETracker, STrack, mot_detector, \
    matching, BaseTrack, TrackState, joint_stracks, sub_stracks, remove_duplicate_stracks
from substractor import SubstracktorDetector
from utils import visualization as vis
from utils.log import logger
import time
import numpy



class SubstractorTracker(JDETracker):
    def __init__(self, opt, frame_rate=30):
        JDETracker.__init__(self, opt, frame_rate=frame_rate)
        self.substractor = SubstracktorDetector()

    def time_lost(self, track):
        return self.frame_id - track.end_frame

    def update(self, img0, im_blob):
        detections = mot_detector(img0, im_blob, self.model,
                                  self.opt.conf_thres,
                                  self.opt.nms_thres,
                                  self.opt.img_size)
        self.frame_id += 1
        detections = [d for d in detections if numpy.prod(d.tlwh[-2:]) > 250]
        frame = vis.plot_tracking(img0,
                          [act.tlwh for act in detections],
                          [act.track_id for act in detections],
                          frame_id=self.frame_id,
                              fps=1.)
        import cv2
        cv2.imshow('detections', frame)
        cv2.waitKey(1)
        output_stracks, activated, lost, tracked = self._update(detections)
        lost_frame = vis.plot_tracking(img0,
                          [act.tlwh for act in lost],
                          [act.track_id for act in lost],
                          frame_id=self.frame_id,
                              fps=1.)
        tracked = vis.plot_tracking(img0,
                          [act.tlwh for act in output_stracks],
                          [act.track_id for act in output_stracks],
                          frame_id=self.frame_id,
                              fps=1.)
        cv2.imshow('output_stracks', tracked)
        cv2.imshow('lost', lost_frame)
        cv2.waitKey(1)
        # detections1 = self.substractor.detect(img0)
        # output_stracks, activated, lost = self._update(detections1)

        return output_stracks + [t for t in lost if self.time_lost(t) < 10]

    def _update(self, detections):
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        t1 = time.time()
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        if self.frame_id == 111:
            import pdb;pdb.set_trace()
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        self.merge(removed_stracks, tracked_stracks)
        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        if self.frame_id != self.predicted_frame_id:
            STrack.multi_predict(strack_pool)
            self.predicted_frame_id = self.frame_id
        matches, u_detection, u_track = self.match(detections, strack_pool)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        #r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.6)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.9)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return output_stracks, activated_starcks, self.lost_stracks, self.tracked_stracks

    def merge(self, removed_stracks, tracked_stracks):
        tr1 = [t for t in tracked_stracks if 10 < self.time_lost(t)]
        tr2 = [t for t in self.lost_stracks if 10 < self.time_lost(t)]
        matched = self.merge_tracks(tr1, tr2)
        to_del = []
        for idx2, idx1 in matched:
            import pdb;pdb.set_trace()
            t1 = tr1[idx1]
            t2 = tr2[idx2]
            print('{0} merged with {1}'.format(t1.track_id, t2.track_id))
            id_react, id_del = min(t2.track_id, t1.track_id), max(t2.track_id, t1.track_id)
            t2.re_activate(t1, frame_id=self.frame_id)
            t2.track_id = id_react
            to_del.append(id_del)
        for t in tracked_stracks:
            if t.track_id in to_del:
                t.mark_removed()
                removed_stracks.append(t)
                tracked_stracks.remove(t)

    def match(self, detections, strack_pool, thresh=0.7):
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=thresh)
        return matches, u_detection, u_track

    def merge_tracks(self, tracks1, tracks2):
        matches, u_detection, u_track = self.match(tracks1, tracks2, thresh=0.6)
        return matches

