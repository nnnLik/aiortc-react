import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from operator import le

import cv2 as cv
import numpy as np
import mediapipe as mp

from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
import aiohttp_cors
from aiohttp import web

from model import KeyPointClassifier
from model import PointHistoryClassifier


ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )


def find_middle(a, b):
    return list((np.array(a) + np.array(b)) / 2)


def append_word(sentence, word):
    if sentence == []: 
        sentence.append(word)
    if sentence[-1] != word:
        sentence.append(word)
    return sentence
         

def get_id(landmark_list, debug_image, point_history, 
            keypoint_classifier, history_length,
            point_history_classifier, finger_gesture_history, side="right"):
        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = pre_process_landmark(
            landmark_list)
        if side == "left":
            b = [0] * len(pre_processed_landmark_list)
            for i, el in enumerate(pre_processed_landmark_list):
                b[i] = -el if i % 2 == 0 else el
            pre_processed_landmark_list = b
        pre_processed_point_history_list = pre_process_point_history(
            debug_image, point_history)
        # Hand sign classification
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        # Point gesture
        point_history.append(landmark_list[0])
        
        # Finger gesture classification
        finger_gesture_id = 0
        point_history_len = len(pre_processed_point_history_list)
        if point_history_len == (history_length * 2):
            finger_gesture_id = point_history_classifier(
                pre_processed_point_history_list)
        
        # Calculates the gesture IDs in the latest detection
        finger_gesture_history.append(finger_gesture_id)
        most_common_fg_id = Counter(
            finger_gesture_history).most_common()
        hand_gest_id = most_common_fg_id[0][0]

        return hand_sign_id, hand_gest_id
        


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_info_text(image, hand_sign_text):

    if hand_sign_text != "":        
        cv.putText(image, "SIGN:" + ','.join(hand_sign_text), (50,50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv.LINE_AA)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (100, 100),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (100, 100),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def solve(bl, tr, p,side) :
    if side=="left":
        if (p[0] >= bl[0] and p[0] <= tr[0] and p[1] <= bl[1] and p[1] >= tr[1]) :
            return True
        else :
            return False
    if side=="right" :
        if (p[0] <= bl[0] and p[0] >= tr[0] and p[1] <= bl[1] and p[1] >= tr[1]) :
            return True
        else :
            return False

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, sentence=[], hist_id=[],
                    hist_id_left=[], length=16):
        super().__init__()
        self.track = track
        self.sentence = sentence
        self.hist_id = hist_id
        self.hist_id_left = hist_id_left
        self.history_length = length
        self.point_history_right = deque(maxlen=self.history_length)
        self.point_history_left = deque(maxlen=self.history_length)

    async def recv(self):

        # Finger gesture history ################################################
        finger_gesture_history_right = deque(maxlen=self.history_length)
        finger_gesture_history_left = deque(maxlen=self.history_length)
        #########################################################################

        frame = await self.track.recv()
        image = frame.to_ndarray(format="bgr24")
        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

            # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                    encoding='utf-8-sig') as f:
                keypoint_classifier_labels = csv.reader(f)
                keypoint_classifier_labels = [
                    row[0] for row in keypoint_classifier_labels
                ]
        with open(
                    'model/point_history_classifier/point_history_classifier_label.csv',
                    encoding='utf-8-sig') as f:
                point_history_classifier_labels = csv.reader(f)
                point_history_classifier_labels = [
                    row[0] for row in point_history_classifier_labels
                ]

        
        # hist_gest = []
        # hist_id = []
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = cv.flip(image, 1) 
        results = holistic.process(image)
       
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.pose_landmarks is not None:
            pose_landmarks = results.pose_landmarks.landmark
            left_shoulder = [pose_landmarks[11].x, pose_landmarks[11].y]
            right_shoulder = [pose_landmarks[12].x, pose_landmarks[12].y]
            right_eye= [pose_landmarks[4].x, pose_landmarks[4].y]
            left_eye= [pose_landmarks[1].x, pose_landmarks[1].y]
            nose =  [pose_landmarks[0].x, pose_landmarks[0].y]
            mid_shoulders = find_middle(left_shoulder, right_shoulder)
            right_hip = [pose_landmarks[24].x, pose_landmarks[24].y]
            left_hip = [pose_landmarks[23].x, pose_landmarks[23].y]
            mid_hip=find_middle(left_hip,right_hip)
            middle=find_middle(mid_hip,mid_shoulders)
            middle_eye=find_middle(left_eye,right_eye)

            handedness = "right"
            #  ####################################################################
            #  Right hand
            if results.left_hand_landmarks is not None:
                
                hand_landmarks = results.left_hand_landmarks.landmark
                right_wrist = [hand_landmarks[0].x, hand_landmarks[0].y]
                right_pointer = [hand_landmarks[8].x, hand_landmarks[8].y]
                # Bounding box calculation
                handedness = "right"
                
                # Landmark calculation
                landmark_list = calc_landmark_list(image, hand_landmarks)

                right_hand_sign_id, right_hand_gest_id = get_id(
                            landmark_list, image, self.point_history_right, 
                            keypoint_classifier, self.history_length,
                            point_history_classifier, finger_gesture_history_right,
                            side=handedness
                )
                image = draw_landmarks(image, landmark_list)
                
            else:
                self.point_history_right.append([0, 0])
                right_hand_sign_id, right_hand_gest_id = -1, -1
            ############################################################################
            # Left hand
            if results.right_hand_landmarks is not None:
                hand_landmarks = results.right_hand_landmarks.landmark
                left_wrist = [hand_landmarks[0].x, hand_landmarks[0].y]
                # Bounding box calculation
                handedness = "left"
                landmark_list = calc_landmark_list(image, hand_landmarks)

                left_hand_sign_id, left_hand_gest_id = get_id(
                            landmark_list, image, self.point_history_left, 
                            keypoint_classifier, self.history_length,
                            point_history_classifier, finger_gesture_history_left,
                            side=handedness
                )
                image = draw_landmarks(image, landmark_list)
            
            else:
                self.point_history_left.append([0, 0])
                left_hand_sign_id, left_hand_gest_id = -1, -1
            
            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]
            
            
            self.hist_id.append(right_hand_sign_id)
            if len(self.hist_id) > 5:
                    self.hist_id=self.hist_id[-5:]
            if len(set(self.hist_id)) == 1:
                right_hand_gest_id = self.hist_id[0]
            elif len(set(self.hist_id)) > 2:
                right_hand_gest_id = -1
            else:
                right_hand_gest_id =  Counter(self.hist_id).most_common()[0][0]


            self.hist_id_left.append(right_hand_sign_id)
            if len(self.hist_id_left) > 5:
                    self.hist_id_left=self.hist_id_left[-5:]
            if len(set(self.hist_id_left)) == 1:
                left_hand_gest_id = self.hist_id_left[0]
            elif len(set(self.hist_id_left)) > 2:
                left_hand_gest_id = -1
            else:
                left_hand_gest_id =  Counter(self.hist_id_left).most_common()[0][0]

            
                    
            if right_hand_sign_id == 7  and right_hand_gest_id in [1, 8, 0, 4] \
                and solve(middle, left_shoulder, right_wrist, "left"):
                        append_word(self.sentence, 'pain')
            elif left_hand_sign_id == 7  and left_hand_gest_id in [5, 3, 0]\
                and solve(middle, right_shoulder, left_wrist, "right"):
                append_word(self.sentence, 'pain')
            elif right_hand_sign_id in [6, 4]  and right_hand_gest_id not in [4, 7, 8]\
                and solve(left_shoulder, middle_eye, right_wrist, "right") and \
                left_hand_sign_id == -1:
                append_word(self.sentence, 'hello')
            elif left_hand_sign_id in [6, 4]  and left_hand_gest_id not in [4, 8]\
                and solve(right_shoulder, middle_eye, left_wrist, "left"):
                append_word(self.sentence, 'hello')
            elif right_hand_sign_id == 9 and right_hand_gest_id in [0, 2, 6]\
                and right_pointer[1] < right_shoulder[1] and right_pointer[1] > nose[1]\
                    and left_hand_sign_id == 9 and left_hand_gest_id in [0, 2, 7]:
                    append_word(self.sentence, 'thanks')
            elif right_hand_sign_id == 11\
                and left_hand_sign_id == 11 and right_wrist[1] > right_shoulder[1]\
                    and left_wrist[1] > left_shoulder[1]:
                append_word(self.sentence, 'depression')
            elif right_hand_sign_id == 12 and left_hand_sign_id == 12\
                and not solve(middle, left_shoulder, right_wrist, "left"):
                append_word(self.sentence, 'health')
            elif right_hand_sign_id == 10 and right_hand_gest_id in [2, 7, 8]\
                and solve(middle, left_shoulder, right_wrist, "left"):
                append_word(self.sentence, 'depression')
            elif (right_hand_sign_id == 1 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]) \
                    or (left_hand_sign_id == 1 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence, 'a')
            elif right_hand_sign_id == 0 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 0 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence, 'b')
            elif right_hand_sign_id == 3 and right_hand_gest_id == 0\
            and right_wrist[1] > right_shoulder[1]\
                or (left_hand_sign_id == 3 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence, 'c')
            elif right_hand_sign_id == 4 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 4 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence, 'e')
            elif right_hand_sign_id == 5 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 5 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence, 'i')
            elif right_hand_sign_id == 3 and right_hand_gest_id == 2\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 3 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence, 'ch')
            else:
                pass


            image = draw_info_text(
                    image,
                    self.sentence)
    
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        pc.addTrack(
            VideoTransformTrack(
                relay.subscribe(track)
            )
        )
        if args.record_to:
            recorder.addTrack(relay.subscribe(track))

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })

    for route in list(app.router.routes()):
        cors.add(route)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )