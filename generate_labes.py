import argparse
import os
import pandas as pd
import cv2


names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']


min_confidence = 0.5


def process(fname, images_dir, txt_dir):
    df = pd.read_csv(
        fname,
        sep=',',
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=names,
        engine='python'
    )

    images = sorted(os.listdir(images_dir))

    # Remove all rows without sufficient confidence
    df = df[df['Confidence'] >= min_confidence]
    result_dir = './tmp/'


    img0 = cv2.imread(os.path.join(images_dir, images[0]))
    # hope that all images have the same size
    img_h, img_w = img0.shape[:2]

    frames = set(x[0] for x in df.index.values)
    lines_template = '0 {Id} {X} {Y} {W} {H}'
    all_ids = set(x[1] for x in df.index.values)
    id_map = dict()
    i = 0
    for old_id in all_ids:
        id_map[old_id] = i
        i += 1

    for i in range(1, len(images) + 1):

        frame_id = i
        result_txt = os.path.join(txt_dir, os.path.splitext(images[frame_id - 1])[0] + '.txt')

        if frame_id not in frames:
            with open(result_txt, 'wt') as f:
                continue

        assert str(frame_id) + '.' in result_txt
        frame_data = df.xs(frame_id, level='FrameId')
        # expected format = [class] [identity] [x_center] [y_center] [width] [height]
        # class = 0

        lines = []

        for id in frame_data.index:
            item = frame_data.xs(id)
            # compute center
            w = item.Width
            h = item.Height
            x = max(0.0, item.X + w / 2.0)
            y = max(0.0, item.Y + h / 2.0)
            w /= img_w
            h /= img_h
            x /= img_w
            y /= img_h
            lines.append(lines_template.format(**{k: round(v, 6) for (k, v) in dict(Id=id_map[id], X=x, Y=y, W=w, H=h).items()}) + '\n')

        with open(result_txt, 'wt') as f:
            f.writelines(lines)

def main():
    parser = argparse.ArgumentParser(description='convert ground truth file to file per frame')
    parser.add_argument('--gt', type=str,
                    help='path to ground truth file')
    parser.add_argument('--images', type=str,
                        help='path to images')
    parser.add_argument('--txt-dir', type=str,
                        help='output directory')

    args = parser.parse_args()
    process(args.gt, args.images, args.txt_dir)


if __name__ == '__main__':
    main()
