import argparse
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from moviepy.video.io.bindings import mplfig_to_npimage

from narya.utils.vizualization import draw_pitch

LOGGER = logging.getLogger(__name__)


def find_contours(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)[0][0]
    return contours


def find_visible_area(h, template_size, image_size):
    """
    template->imageのhomographyからnarya座標におけるvisible areaを算出
    """

    h_inv = np.linalg.inv(np.array(h))
    template_h, template_w = template_size
    image_h, image_w = image_size

    im = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255
    im = cv2.warpPerspective(im, h_inv, dsize=(template_w, template_h))
    contours = find_contours(im)
    xs = [c[0][0] * 100 / template_w for c in contours]
    ys = [(template_h - c[0][1]) * 100 / template_h for c in contours]
    return xs, ys


def draw_image(homo_rec, detect_df):
    xs, ys = find_visible_area(
        homo_rec['homography'], homo_rec['template_size'], homo_rec['image_size'])

    fig, ax = draw_pitch(pitch_color='gray')
    ax.fill(xs, ys, c='green', alpha=0.30)

    for _, row in detect_df.iterrows():
        if row['cid'] == 1:
            ax.add_artist(
                Ellipse(
                    (row['x_plot'], row['y_plot']),
                    3 / 105 * 100,
                    3 / 68 * 100,
                    edgecolor='white',
                    facecolor='red'
                )
            )

    image = mplfig_to_npimage(fig)
    plt.close()
    return image


def warp_detect_point(h_inv, x, y, image_size, template_size, detect_size):
    """
    warp detected point to plot coordinate
    """

    image_h, image_w = image_size
    template_h, template_w = template_size
    detect_h, detect_w = detect_size

    x_image = x * image_w / detect_w
    y_image = y * image_h / detect_h
    x_warp, y_warp, z_warp = h_inv @ np.array([x_image, y_image, 1])
    x_template = x_warp / z_warp
    y_template = y_warp / z_warp
    x_plot = x_template * 100 / template_w
    y_plot = (template_h - y_template) * 100 / template_h

    return x_plot, y_plot


def warp_detect_df(detect_df, homo_rec_map):
    detect_df['x'] = (detect_df['x1'] + detect_df['x2']) / 2
    detect_df['y'] = detect_df[['y1', 'y2']].max(axis=1)

    xs = []
    ys = []
    for _, row in detect_df.iterrows():
        h_inv = homo_rec_map[row['image']]['h_inv']
        x, y = warp_detect_point(h_inv, row['x'], row['y'], (720, 1280), (74, 115), (512, 512))
        xs.append(x)
        ys.append(y)

    detect_df['x_plot'] = xs
    detect_df['y_plot'] = ys

    return detect_df


def main():
    with open(args.homo, 'r') as f:
        homo_data = json.load(f)
    LOGGER.info(f'loaded {len(homo_data)} homography from {args.homo}')

    homo_rec_map = dict()
    for homo_rec in homo_data:
        homo_rec['h_inv'] = np.linalg.inv(np.array(homo_rec['homography']))
        homo_rec_map[homo_rec['image_id']] = homo_rec

    detect_df = pd.read_csv(args.detect)
    detect_df = warp_detect_df(detect_df, homo_rec_map)

    for homo_rec in homo_data:
        image_id = homo_rec['image_id']
        frame_df = detect_df[detect_df['image'] == image_id]
        image = draw_image(homo_rec, frame_df)
        out_fp = f'{args.out}/{image_id}.jpg'
        cv2.imwrite(out_fp, image)
        LOGGER.info(f'saved {out_fp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--homo', help='SCCvSDで計算したhomography(.json)', required=True)
    parser.add_argument('--detect', help='detection.pyで計算したbbox(.csv)')
    parser.add_argument('--out', help='プロット結果のディレクトリ', required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    main()
