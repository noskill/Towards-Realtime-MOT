import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

import torch
from tracker.multitracker import STrack, JDETracker
from multidetector import SubstractorTracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from utils.utils import *


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    # tracker = JDETracker(opt, frame_rate=frame_rate)
    tracker = SubstractorTracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    manualMode = True
    import time
    for path, img, img0 in dataloader:
        frame_id += 1
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))
        # if frame_id < 700:
        #     continue
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        ''' Step 1: Network forward, get detections & embeddings'''
        online_targets = tracker.update(img0, blob)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > opt.min_box_area and (t.tracklet_len > 2):
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
            wait_time = 0 if manualMode else 1
            k: int = cv2.waitKey(wait_time)
            if (k == ord('q')):
                break
            elif k == ord('m') or k == ord('M'):
                manualMode = not manualMode
                if manualMode:
                    cv2.waitKey(0)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo', 
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.DEBUG)
    result_root = os.path.join(".", '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'
    # Read config
    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(".", '..','outputs', exp_name, seq) if save_images or save_videos else None

        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadVideo("/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/set004_2020-02-11/fragment1.mp4")
        dataloader = datasets.LoadVideo("/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/fragments/meeting/meeting_set005_1:26:20-1:26:20.mp4")
        # dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read() 
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir="./vidos", show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
    opt = parser.parse_args()
    print(opt, end='\n\n')



    data_root = "/mnt/fileserver/shared/datasets/MOT/MOT17Det/test/"
    seqs_str = 'MOT17-03 MOT17-06  MOT17-01  MOT17-07  MOT17-08  MOT17-12 MOT17-14 '

    data_root = '/mnt/fileserver/shared/datasets/streetscene/Test/'
    seqs_str = 'test001_mot 1'


    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.weights.split('/')[-2],
         show_image=True,
         save_images=opt.save_images, 
         save_videos=opt.save_videos)

