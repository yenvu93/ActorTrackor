Evaluation of Actors Tracking Method in Case of Variant Appearance Introduction

A significant obstacle is unseen films that can not recognize characters in variant appearances. This paper considers using text annotations in form of subtitles and scripts to extract an initial prediction of who appears in the video and when they appear. It is motivated to carry out this challenge by building a deep convolutional neural network to detect pedestrians and automatically detect who is speaking by face detection and lip emotion methodology. Then labeling the correct name of each speaker in the frame using textual semantic cues. Moreover, we also develop a set of results containing actor identification, frame number, and image location which is helpfully extracted the according to actors.

Installation at [Install.md](https://github.com/yenvu93/ActorTrackor/blob/main/Install.md)


-----------------------------------------------------------------------------------------
Trainning Person Dectection

+ Running convert MovieNet Dataset to COCO dataset

python3 ./tools/movienet2coco.py -i ./data/ -o ./data/actorTracker/annotations --version 2019

+ Running train model

copy person_detection_model stuffs to approproate MMdetection folder (Checking https://mmdetection.readthedocs.io/en/latest/ )

Run:

cd mmdectection

python3 tools/train.py ./person_detection_model/person_detection_cascade_rcnn_r50_dc5_movienet_movienet.py

-------------------------------------------------------------------------------------------

Run Actor tracking in video:

cd actortracker/demos

python3 actor_tracker.py -p facial-landmarks/shape_predictor.dat -v demo.mp4
