import os
import cv2
import pandas as pd
from utils import visualization as vis


def show_box():
    import pdb;pdb.set_trace()
    path = '/mnt/fileserver/shared/datasets/MOT/MOT17Det/train/MOT17-04/'
    fname = os.path.join(path, 'gt', 'gt.txt')
    img_dir = os.path.join(path, 'img1')
    files = os.listdir(img_dir)
    sep = r'\s+|\t+|,'
    min_confidence = 0.5

    df = pd.read_csv(
        fname,
        sep=sep,
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],
        engine='python'
    )

    # Account for matlab convention.
    df[['X', 'Y']] -= (1, 1)

    # Removed trailing column
    del df['unused']

    # Remove all rows without sufficient confidence
    df = df[df['Confidence'] >= min_confidence]
    import pdb;pdb.set_trace()
    for i, f in enumerate(files):
        frame_id = i + 1
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        cv2.imshow("img", img)


        frame_data = df.xs(frame_id, level='FrameId')
        tlwh = frame_data[['X', 'Y', 'Width', 'Height']].to_numpy()
        track_id = frame_data.index.values
        frame = vis.plot_tracking(img,
                          tlwh,
                          track_id,
                          frame_id=frame_id,
                              fps=1.)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)


show_box()