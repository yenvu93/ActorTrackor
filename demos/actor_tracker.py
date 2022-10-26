import argparse
from threading import activeCount
from textual_parser import get_label, get_textuals
import dlib
import cv2
import numpy as np
from video_process import processing_video
import os.path as osp
import mmcv
import os
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=False)
args = vars(ap.parse_args())

def draw_person(IMAGE, persons, category):

    img = IMAGE.copy()
    img = cv2.rectangle(img, (int(persons[0]), int(persons[1])),
                        (int(persons[2]), int(persons[3])),
                        (0, 255, 0), 2)
    img = cv2.putText(
        img, category,
        (int(persons[0] + 10), int(persons[1] + 30)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


""" Function for test a dynamic wrapping time, using MovieNet annotations 
    Args:
        ann_dir (A absolute path of a dataset)
    Raises:
        ValueError: Thrown when specifying a negative path of required files
    """
def convert_vis(ann_dir, mode='test'):
    #List of information of videos such as id, width, height, frame, durations
    video_infos = mmcv.load(osp.join(ann_dir, '{}.json'.format("video_info")))

    #List of specified videos for train, valid, and test
    video_list = mmcv.load(osp.join(ann_dir, '{}.json'.format("video_list")))
    # os.mkdir(path)
    video_Ids = video_list[mode]
    for video_id in tqdm(video_Ids):
        video_meta = mmcv.load(osp.join(ann_dir + "/meta/", '{}.json'.format(video_id)))

        video_annotations = mmcv.load(osp.join(ann_dir + "/annotations/", '{}.json'.format(video_id)))

        script=osp.join(ann_dir + "/actortracker/script/", '{}.html'.format(video_id))
        subtitle=osp.join(ann_dir + "/actortracker/subtitle/", '{}.srt'.format(video_id))
        cast_annotations = video_annotations['cast']
        subtitle_annos = video_annotations['story']
        cast_infos = video_meta['cast']

        import pickle
        labels_dict = pickle.load(open(osp.join(input + "wrapping_name/", '{}.pickle'.format(video_id)), "rb"))

        predict = []
        actual = []

        actual_cast = []
        for frame_id in range(len(video_annotations['cast'])):
            if frame_id == 10:
                break
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
            pid = info['pid']
            truth_name = [n for n in cast_infos if n['id'] == pid]
            print("---------------------------")
            print(truth_name)
            print(image_path)

        for shots in subtitle_annos:
            for subtitle in shots['subtitle']: 
                shot_idx = subtitle['shot']
                list_shots = [ca for ca in cast_annotations if ca['shot_idx'] == shot_idx]
                predict_cast = np.array([id for id in labels_dict if id[1] == shot_idx ])
                hi = predict_cast[:,2]
                predict_cast = np.unique(hi)
                actual_cast = []
                for shot in list_shots:
                    id = shot['pid']
                    name = [n for n in cast_infos if n['id'] == id]
                    if name:
                        actual_cast.append(name[0]['character'].lower())
                actual_cast = np.array(np.unique(actual_cast))
                for pre in predict_cast:
                    if len(actual_cast) == 0:
                        actual.append(0)
                        predict.append(1)
                        continue
                    ac = [ i for i in actual_cast if pre.lower() == i ]
                    if ac:
                        predict.append(1)
                        actual.append(1)
                    else:
                        predict.append(0)
                        actual.append(1)

""" Function for getting confusion matrix
    Args:
        input (A absolute path of a dataset)
        video_id (video id as MovieNet dataset ID)
    Raises:
        ValueError: Thrown when specifying a negative path of required files
    """
def confusionMatrix(input, video_id):

    predict, actual = [], []
    text_file = open(osp.join(input + "out/", '{}_predict.txt'.format(video_id)), "r")
    predict = text_file.readlines()

    text_file.close()

    text_file = open(osp.join(input + "out/", '{}_actual.txt'.format(video_id)), "r")
    actual = text_file.readlines()

    text_file.close()

    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(actual, predict))

    matrix = confusion_matrix(actual,predict )
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pretty_confusion_matrix import pp_matrix
    classes = ['BILLY','CYNTHIA','DAD','JOSH','MOM', 'speaker_unknown']
    # class = ['BILLY','CYNTHIA','JOSH','speaker_unknown']
    import pandas as pd
    df_cm = pd.DataFrame(matrix, index=classes, columns=classes)
    cmap = 'PuRd'
    pp_matrix(df_cm, cmap=cmap)

    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d') # font size

    plt.show()

def main():
    input="/home/yen/movienet/data/actortracker/"
    if args["video"]:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args["shape_predictor"])
        processing_video(input, args["video"], detector, predictor, "tt0248667")
        
    else:
        convert_vis(input, 'test')
        confusionMatrix(input, "tt0248667")

if __name__ == '__main__':
    main()
   


