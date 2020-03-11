import os
import pandas as pd
import cv2


names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']


fname = '/mnt/fileserver/shared/datasets/MOT/MOT17Det/train/MOT17-04/gt/gt.txt'
images_dir = '/mnt/fileserver/shared/datasets/MOT/MOT17Det/train/MOT17-04/img1/'
txt_dir = '/mnt/fileserver/shared/datasets/MOT/MOT17Det/train/MOT17-04/gt/'

images = os.listdir(images_dir)
min_confidence = 0.5
df = pd.read_csv(
        fname,
        sep=',',
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=names,
        engine='python'
    )
# Remove all rows without sufficient confidence
df = df[df['Confidence'] >= min_confidence]
result_dir = './tmp/'


img0 = cv2.imread(os.path.join(images_dir, images[0]))
# hope that all images have the same size
img_h, img_w = img0.shape[:2]

frames = set(x[0] for x in df.index.values)
lines_template = '0 {Id} {X} {Y} {W} {H}'

for frame_id in range(len(images)):
    result_txt = os.path.join(txt_dir, os.path.splitext(images[frame_id])[0] + '.txt')
    if frame_id not in frames:
        with open(result_txt, 'wt') as f:
            continue
    frame_data = df.xs(frame_id, level='FrameId')
    # expected format = [class] [identity] [x_center] [y_center] [width] [height]
    # class = 0

    lines = []

    for id in frame_data.index:
        item = frame_data.xs(id)
        # compute center
        w = item.Width
        h = item.Height
        x = item.X + w / 2.0
        y = item.Y + h / 2.0
        w /= img_w
        h /= img_h
        x /= img_w
        y /= img_h
        lines.append(lines_template.format(**{k: round(v, 6) for (k, v) in dict(Id=id, X=x, Y=y, W=w, H=h).items()}) + '\n')

    with open(result_txt, 'wt') as f:
        f.writelines(lines)


