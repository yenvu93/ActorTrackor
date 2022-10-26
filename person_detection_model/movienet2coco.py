# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from operator import mod
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='MovieNetto COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of MovieNetannotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    parser.add_argument(
        '--version',
        choices=['2019', '2021'],
        help='The version of MovieNetDataset',
    )
    return parser.parse_args()

import cv2
def draw_person(IMAGE, persons, category):
    # read image
    img = IMAGE.copy()
    img = cv2.rectangle(img, (int(persons[0]), int(persons[1])),
                        (int(persons[2]), int(persons[3])),
                        (0, 255, 0), 2)
    img = cv2.putText(
        img, category,
        (int(persons[0] + 10), int(persons[1] + 30)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img

def convert_vis(ann_dir, save_dir, dataset_version, mode='train'):
    """Convert MovieNetdataset in COCO style.

    Args:
        ann_dir (str): The path of MovieNetdataset.
        save_dir (str): The path to save `VIS`.
        dataset_version (str): The version of dataset. Options are '2019',
            '2021'.
        mode (str): Convert train dataset or validation dataset or test
            dataset. Options are 'train', 'valid', 'test'. Default: 'train'.
    """
    assert dataset_version in ['2019', '2021']
    assert mode in ['train', 'val', 'test']
    VIS = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    obj_num_classes = dict()
    category_index = 0

    #List of information of videos such as id, width, height, frame, durations
    video_infos = mmcv.load(osp.join(ann_dir, '{}.json'.format("video_info")))

    #List of specified videos for train, valid, and test
    video_list = mmcv.load(osp.join(ann_dir, '{}.json'.format("video_list")))

    has_annotations = mode == 'train' or mode == 'val'

    if has_annotations:
        vid_to_anns = defaultdict(list)
        for id in video_list[mode]:
            video_annotations = mmcv.load(osp.join(ann_dir + "/annotations/", '{}.json'.format(id)))
            vid_to_anns[id].append(video_annotations['cast'])

    video_Ids = video_list[mode]
    for video_id in tqdm(video_Ids):
        video_meta = mmcv.load(osp.join(ann_dir + "/meta/", '{}.json'.format(video_id)))

        video_annotations = mmcv.load(osp.join(ann_dir + "/annotations/", '{}.json'.format(video_id)))
        frame_annotations = video_annotations['cast']

        # Video processing
        video_info = video_infos[video_id]
        video_name = video_meta['title']
        video = dict(
            id=video_id,
            name=video_name,
            width=frame_annotations[0]['resolution'][0],
            height=frame_annotations[0]['resolution'][1])
        VIS['videos'].append(video)

        #Images processing
        #List of annotated videos such as cast, bbox

        if has_annotations:
            ann_infos_in_video = vid_to_anns[video_id]
            instance_id_maps = dict()

        for frame_id in range(len(video_annotations['cast'])):
            info = video_annotations['cast'][frame_id]
            id = str(info['id'][:9])
            shot_idx = '{0:04d}'.format(info['shot_idx'])
            file_name= id + "/shot_" + str(shot_idx) + "_img_" + str(info['img_idx']) + ".jpg"
            import cv2
            image_path = ann_dir + "actortracker/" + mode
            image_path = os.path.join(image_path, file_name)
            # print(image_path)
            img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
            if img is None:
                continue
                print("------------------------------")
            h, w, _ = img.shape
            image = dict(
                file_name=file_name,
                height=h,
                width=w,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=id)
            VIS['images'].append(image)

            if mode == 'test':
                if records['img_id'] > 5:
                    break
        
            if has_annotations:
                bbox = info['body']['bbox']
                if bbox is None:
                    continue

                category = info['pid']

                if category != "null":
                    category_id=int(categories[0]['id'])

        
                track_id = info['id']

                if track_id in instance_id_maps:
                    instance_id = instance_id_maps[track_id]
                else:
                    instance_id = records['global_instance_id']
                    records['global_instance_id'] += 1
                    instance_id_maps[track_id] = instance_id

                x1, y1, x2, y2 = [bbox[0], bbox[1], bbox[2], bbox[3]]
                w = x2 - x1 
                h = y2 - y1
                bbox=[x1, y1, w, h]

                ann = dict(
                    id=records['ann_id'],
                    video_id=video_id,
                    image_id=records['img_id'],
                    category_id=category_id,
                    instance_id=instance_id,
                    bbox=bbox,
                    area=w*h,
                    iscrowd=False)

                if category_id not in obj_num_classes:
                    obj_num_classes[category_id] = 1
                else:
                    obj_num_classes[category_id] += 1

                VIS['annotations'].append(ann)
                records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1
    VIS['categories'] = copy.deepcopy(categories)

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(VIS,
              osp.join(save_dir, f'youtube_vis_{dataset_version}_{mode}.json'))
    print(f'-----YouTube VIS {dataset_version} {mode}------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["img_id"]- 1} images')
    if has_annotations:
        print(f'{records["ann_id"] - 1} objects')
        print(f'{records["global_instance_id"] - 1} instances')
    print('-----------------------')
    if has_annotations:
        for i in range(1, len(VIS['categories'])):
            class_name = VIS['categories'][i - 1]['name']
            print(f'\'{class_name}\',')

def main():
    args = parse_args()
    for sub_set in ['train', 'val', 'test']:
        convert_vis(args.input, args.output, args.version, sub_set)

# categories = []
categories =  [{"id": 0, "name": "pedestrian"}]
if __name__ == '__main__':
    main()
