import glob

import argparse
import cv2
import logging
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from narya.models.gluon_models import TrackerModel

LOGGER = logging.getLogger(__name__)


def read_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def initialize_tracker_model():
    tracker_model = TrackerModel(pretrained=True, backbone='ssd_512_resnet50_v1_coco')
    WEIGHTS_PATH = (
        "https://storage.googleapis.com/narya-bucket-1/models/player_tracker.params"
    )
    WEIGHTS_NAME = "player_tracker.params"
    WEIGHTS_TOTAR = False
    checkpoints = tf.keras.utils.get_file(
        WEIGHTS_NAME, WEIGHTS_PATH, WEIGHTS_TOTAR,
    )
    tracker_model.load_weights(checkpoints)
    return tracker_model


def main():
    img_paths = sorted(glob.glob(f'{args.input}/*.jpg'))
    LOGGER.info(f'found {len(img_paths)} images in {args.input}')

    tracker_model = initialize_tracker_model()
    LOGGER.info(f'initialized TrackerModel')

    records = []
    for image_path in tqdm(img_paths):
        image = read_image(image_path)
        cid, score, bbox = tracker_model(image, split_size=512)
        mask = score[0, :, 0] > 0.5
        cid, score, bbox = cid[0, mask, 0], score[0, mask, 0], bbox[0, mask]

        recs = []
        for i in range(len(cid)):
            rec = dict()
            rec['image'] = Path(image_path).stem
            rec['cid'] = int(cid[i])
            rec['score'] = score[i]
            rec['x1'], rec['y1'], rec['x2'], rec['y2'] = bbox[i]
            recs.append(rec)

        num_balls = [rec['cid'] == 0 for rec in recs]
        num_players = [rec['cid'] == 1 for rec in recs]
        LOGGER.debug(f'detected {num_balls} balls and {num_players} players from {image_path}')
        records += recs

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out, index=False)
    LOGGER.info(f'saved {len(out_df)} records in {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='入力画像フォルダ', required=True)
    parser.add_argument('--out', help='出力ファイル(json)', required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    main()
