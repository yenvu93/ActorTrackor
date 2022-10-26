# The codes below partially refer to the Automatic-Cast-Listing.
#   ---------------------------------------------------------------
#     [  Github: https://github.com/priyank-trivedi/Automatic-Cast-Listing  ]
#
from imutils.video import FileVideoStream
from imutils import face_utils

import datetime
import math, operator
import imutils
import time
import sys
import cv2
import numpy as np
from threading import Thread
from queue import Queue
from textual_parser import get_textuals
import os.path as osp
from time import mktime
from tqdm import tqdm
import pickle
from movienet.tools import PersonDetector, PersonExtractor
from frame_timecode import FrameTimecode

""" Function for frame-based video processing, using the video framerate
    to detect the face and body of the character then label their's name in each frame.
    Args:
        input (A absolute path of a dataset)
        file (video test path)
        detector &predictor  (existing models, to allow faces detection and extraction)
        video_id (video id as MovieNet dataset ID)
    Raises:
        ValueError: Thrown when specifying a negative path of required files
    """
def processing_video(input, file, detector, predictor, video_id):
    fvs = FileVideoStream(file).start()
    time.sleep(1.0)
    frame_count = 0
    df_pos = 0
    subtitle=osp.join(input + "subtitle/", '{}.srt'.format(video_id))
    script=osp.join(input + "script/", '{}.html'.format(video_id))
    import json
  
    #Processing video list
    f = open(osp.join(input, '{}.json'.format("video_list")))
    video_list = json.load(f)

    labels_dict = []
    try:
        label_path = osp.join(input + "wrapping_name/", '{}.pickle'.format(video_id))
        # print(label_path)
        labels_dict = pickle.load(open(label_path, "rb"))
    except (OSError, IOError) as e:
        print("''''''''''''''''''''''''''''''''''")

        labels_dict = get_textuals(subtitle, script)
        labels_dict = np.array(labels_dict)
        with open(label_path, 'wb') as handle:
            pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cfg = '/home/yen/movienet/model/cascade_rcnn_x101_64x4d_fpn.json'
    weight = '/home/yen/movienet/model/cascade_rcnn_x101_64x4d_fpn.pth'
    detector_person = PersonDetector('rcnn', cfg, weight)

    weight_path = '/home/yen/movienet/model/resnet50_csm.pth'
    extractor = PersonExtractor(weight_path, gpu=0)

    josh_small = cv2.imread('/home/yen/Pictures/josh_small.jpg')
    feat_josh_small = extractor.extract(josh_small)
    feat_josh_small/= np.linalg.norm(feat_josh_small)

    josh_big = cv2.imread('/home/yen/Pictures/josh_big.jpg')
    feat_josh_big= extractor.extract(josh_big)
    feat_josh_big/= np.linalg.norm(feat_josh_big)

    def rmsdiff(im1, im2):
        img = im1-im2
        h,bins = np.histogram(img.ravel(),256,[0,256])
        #h2,bins = np.histogram(im2.ravel(),256,[0,256])
        #h = h1-h2
        sq = (value*(idx**2) for idx, value in enumerate(h))
        sum_of_squares = sum(sq)
        rms = math.sqrt(sum_of_squares/float(im1.shape[0] * im1.shape[1]))
        return rms

    prev_diff_array = None
    flag =0 
    fp = open('temp.txt','a')
    print("Total frame count:", fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(fvs.stream.get(cv2.CAP_PROP_FPS))
    print("FPS:", fps)

    #Get speaker name by comparing frame time and dialogue time
    def getSpeakerName(frame_time, next_iter):

        for iter in range(next_iter, len(labels_dict)):
            start = labels_dict[iter][0]
            end = labels_dict[iter][1]
            start_time =start.strftime('%H:%M:%S.%f')[:-3]
            end_time = end.strftime('%H:%M:%S.%f')[:-3]

            if frame_time > start_time and frame_time < end_time:

                name = labels_dict[iter][2]
                if name:
                    return labels_dict[iter][2], iter
                else:
                    return "speaker_unknown!", iter
        return "speaker_unknown!", next_iter
    
    next_iter = 0
    big=osp.join(input + "demos/out/", '{}.mp4'.format("out_big"))
    cap = cv2.VideoCapture((file))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(big, fourcc, fps, size)
    out_put = []
    while fvs.more():
        frame = fvs.read()

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detection person
        persons = detector_person.detect(frame, show=True, conf_thr=0.5)
        person_imgs = detector_person.crop_person(frame, persons)
        name_person = ""
        diff_array = 0
        name = ""
        for i in range(len(person_imgs)):
            feat = extractor.extract(person_imgs[i])
            feat /= np.linalg.norm(feat)

            predicts = []

            for i in range(len(predicts)):
                img = cv2.rectangle(frame, (int(persons[i, 0]), int(persons[i, 1])),
                                (int(persons[i, 2]), int(persons[i, 3])),
                                (0, 255, 0), 2)
                frame = img.copy()
        #Detect face
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            curr_data = None
            clone = frame.copy()
            cv2.putText(clone, "lips", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            for (x,y) in shape[48:68]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
                roi = frame[y:y + h, x:x + w]
                # print(roi)
                (h, w) = roi.shape[:2]
                if not roi.any():
                    continue
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            if not roi.any():
                continue
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(roi_gray,(100,250), interpolation = cv2.INTER_CUBIC)
            curr_data = res
            if flag==0:
                flag = 1
                prev_data = curr_data
                prev_diff_array = diff_array

            if flag==1:

                diff_array = rmsdiff(prev_data,curr_data)

                thres = abs(prev_diff_array - diff_array)
                fp.write(str(prev_diff_array))
                fp.write(str(diff_array))
    
                name, curr_iter = getSpeakerName(FrameTimecode(frame_count, fps), next_iter)

                next_iter = curr_iter
                if thres>=50:
                    cv2.putText(frame, name , (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, name, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            prev_data = curr_data
            prev_diff_array = diff_array

            subtitle=osp.join(input + "out/", '{}.jpg'.format(frame_count))
            out_put.append([subtitle, name])
            cv2.imwrite(subtitle, frame)
        frame_count += 1
        print(frame_count)
        if frame_count == 10000:
            name_path = label_path = osp.join(input + '{}.pickle'.format(video_id))
            out_put = np.array(out_put)
            with open(name_path, 'wb') as handle:
                pickle.dump(out_put, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # videoWriter.write(frame)
        # cv2.imshow("hi", frame)
        # cv2.waitKey(1)
        # if frame_count == 100:
        #     break

    cv2.destroyAllWindows()
    fvs.stop()