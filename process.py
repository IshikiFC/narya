import argparse
import glob
import json
import logging
from pathlib import Path

import cv2
import imageio
import pandas as pd

from narya.tracker.full_tracker import FootballTracker
from narya.utils.vizualization import make_animation

LOGGER = logging.getLogger(__name__)


def read_template():
    template = cv2.imread('world_cup_template.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = cv2.resize(template, (1280, 720))
    template = template / 255.
    return template


def sample_images(video_path, fps):
    """
    sample images from video by given FPS
    """

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    LOGGER.debug('opened {0}: FPS={1}, length={2}s'.format(video_path, video_fps, video_length))

    img_list = []
    for msec in range(0, int(video_length * 1000), int(1000 / fps)):
        frame_pos = round(video_fps * msec / 1000)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            img_list.append(cv2.resize(frame, (512, 512)))
    cap.release()

    return img_list


def read_processed_images(out_dir):
    img_paths = sorted(glob.glob(out_dir / 'test_*.jpg'))
    img_list = []
    for img_path in img_paths:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_list.append(image)
    return img_list


def to_trajectory_df(trajectories):
    records = []
    for id_, trajectory in trajectories.items():
        for record in trajectory:
            records.append({
                'id': id_,
                'x': record[0] * 100.0 / 320.0,
                'y': 100 - (record[1] * 100.0 / 320.0),
                'frame': record[2]
            })
    df = pd.DataFrame(records)
    df = df.set_index('frame')

    # add fields necessary for visualization
    df['bgcolor'] = 'red'
    df['edgecolor'] = 'white'
    df['team'] = 'unknown'
    df['player'] = [0 if i == -1 else i for i in df['id']]
    df['player_num'] = ''
    df['dx'] = 0
    df['dy'] = 0

    return df


def main():
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    trajectory_path = str(out_dir / 'trajectories.json')
    processed_image_path = str(out_dir / 'test_*.jpg')
    processed_video_path = str(out_dir / 'processed.mp4')
    plot_video_path = str(out_dir / 'plot.mp4')

    if not args.skip_track:
        img_list = sample_images(args.video, args.fps)
        LOGGER.info(f'sampled {len(img_list)} images from {args.video}')

        template = read_template()
        tracker = FootballTracker(frame_rate=args.fps, track_buffer=60)
        trajectories = tracker(img_list, split_size=512, save_tracking_folder=f'{out_dir}/', template=template)
        with open(trajectory_path, 'w') as f:
            json.dump(trajectories, f)
        LOGGER.info(f'saved trajectories to {trajectory_path}')

    img_paths = sorted(glob.glob(processed_image_path))
    with imageio.get_writer(processed_video_path, mode='I', fps=args.fps) as writer:
        for img_path in img_paths:
            image = imageio.imread(img_path)
            writer.append_data(image)
    LOGGER.info(f'saved processed images to {processed_video_path}')

    with open(trajectory_path, 'r') as f:
        trajectories = json.load(f)
    trajectory_df = to_trajectory_df(trajectories)
    plot_clip = make_animation(trajectory_df, voronoi=False, fps=args.fps)
    plot_clip.write_videofile(plot_video_path)
    LOGGER.info(f'saved plot images to {plot_video_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='入力動画ファイル（mp4）', required=True)
    parser.add_argument('--out', help='出力先フォルダ', required=True)
    parser.add_argument('--fps', help='出力動画のFPS', type=float, default=5.0)
    parser.add_argument('--skip_track', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    main()
