#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!py -3.11
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

#sound synthesis
import sounddevice as sd
import threading
import time

samplerate = 1800
volume = 0.2  # master output volume
NUM_VOICES = 4
interp_speed = 0.5  # how fast frequency interpolates

lock = threading.Lock()

# Voice state
frequencies = [440.0] * NUM_VOICES          # current interpolated freq
target_frequencies = [440.0] * NUM_VOICES   # destination freq
phases = [0.0] * NUM_VOICES
active = [False] * NUM_VOICES
volumes = [0.3] * NUM_VOICES  # per-voice volume (0.0 to 1.0)

def audio_callback(outdata, frames, time_info, status):
    global phases, frequencies

    if status:
        print(status)

    wave = np.zeros(frames, dtype=np.float32)

    with lock:
        for i in range(NUM_VOICES):
            if active[i]:
                # Smoothly approach target frequency
                freq_diff = target_frequencies[i] - frequencies[i]
                frequencies[i] += interp_speed * freq_diff

                freq = frequencies[i]
                phase = phases[i]
                vol = volumes[i]

                for n in range(frames):
                    wave[n] += vol * np.sin(phase)
                    phase += 2 * np.pi * freq / samplerate
                    if phase > 2 * np.pi:
                        phase -= 2 * np.pi

                phases[i] = phase

    num_active = sum(active)
    if num_active > 0:
        wave *= (volume / num_active)

    outdata[:] = wave.reshape(-1, 1)

def set_voice_frequency(voice_num, freq):
    with lock:
        if 0 <= voice_num < NUM_VOICES:
            target_frequencies[voice_num] = freq

def toggle_voice(voice_num, on_off):
    with lock:
        if 0 <= voice_num < NUM_VOICES:
            active[voice_num] = on_off

def set_voice_volume(voice_num, vol):
    with lock:
        if 0 <= voice_num < NUM_VOICES:
            volumes[voice_num] = max(0.0, min(vol, 1.0))  # clamp 0–1

def round_to_nearest(value, choices):
    return min(choices, key=lambda x: abs(x - value))

def generate_chromatic_frequencies(min_freq=0, max_freq=540, base_freq=440):
    freqs = []
    for n in range(-57, 50):  # wide enough range to hit 0–540
        f = base_freq * (2 ** (n / 12))
        if min_freq <= f <= max_freq:
            freqs.append(round(f, 2))
    return freqs

chromatic_freqs = generate_chromatic_frequencies()


# Start the stream
stream = sd.OutputStream(
    samplerate=samplerate,
    channels=1,
    callback=audio_callback,
    dtype='float32'
)

#arduino communication
import serial
import queue
#arduino = serial.Serial(port='COM3', baudrate=9600) 
#print("serial open:",arduino.is_open)

serial_queue = queue.Queue()

def serial_writer(ser, q):
    while True:
        try:
            data = q.get(timeout=1)
            if data is None:
                break  # signal to stop
            ser.write(data)
            ser.flush()
        except queue.Empty:
            continue

# Start thread
#threading.Thread(target=serial_writer, args=(arduino, serial_queue), daemon=True).start()



global HEIGHT
HEIGHT=960
global WIDTH
WIDTH=540

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=WIDTH)
    parser.add_argument("--height", help='cap height', type=int, default=HEIGHT)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

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

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    #start audio playing
    stream.start()
    
    left_angle=90

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

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

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list, handedness)


                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                
                #sound control
                if(handedness.classification[0].label[0:]=="Right"):
                    set_voice_frequency(0, round_to_nearest(landmark_list[8][0], chromatic_freqs))
                    set_voice_volume(0, (HEIGHT-2*landmark_list[8][1])/HEIGHT)
                    #print((HEIGHT-landmark_list[8][1])/HEIGHT)
                    if(keypoint_classifier_labels[hand_sign_id]=="Close"):
                        toggle_voice(0, False)
                    else:
                        toggle_voice(0, True)
                        serial_queue.put(("r"+str(int(landmark_list[8][0]/6))+"\n").encode())

                 

                if(handedness.classification[0].label[0:]=="Left"):
                    set_voice_frequency(1, round_to_nearest(landmark_list[8][0], chromatic_freqs))
                    set_voice_volume(1, (HEIGHT-2*landmark_list[8][1])/HEIGHT)
                    if(keypoint_classifier_labels[hand_sign_id]=="Close"):
                        toggle_voice(1, False)
                        toggle_voice(2, False)
                        toggle_voice(3, False)
                        toggle_voice(4, False)
                    else:
                        toggle_voice(1, True)
                        serial_queue.put(("l"+str(int(landmark_list[8][0]/6))+"\n").encode())

                    try:
                        left_angle = np.arctan((landmark_list[1][1]-landmark_list[8][1])/(landmark_list[1][0]-landmark_list[8][0]))
                        left_angle= np.rad2deg(left_angle)
                    except:
                        left_angle=90

                    left_angle=abs(left_angle)

                    #print(left_angle)

                    set_voice_frequency(2, landmark_list[8][0]*1.25)
                    set_voice_frequency(3, landmark_list[8][0]*1.50)
                    set_voice_frequency(4, landmark_list[8][0]*1.89)

                #if arduino.in_waiting > 0:
                 #   line = arduino.readline().decode('utf-8').strip()
                  #  print("Arduino says:", line)

                if(keypoint_classifier_labels[hand_sign_id]=="Open"):
                    if(left_angle>68):
                        #print("base tone")
                        toggle_voice(2, False)
                        toggle_voice(3, False)
                        toggle_voice(4, False)
                    elif(left_angle>45):
                        #print("3rd")
                        toggle_voice(2, True)
                        toggle_voice(3, False)
                        toggle_voice(4, False)
                    elif(left_angle>22):
                        #print("5th")
                        toggle_voice(2, True)
                        toggle_voice(3, True)
                        toggle_voice(4, False)
                    else:
                        #print("7th")
                        toggle_voice(2, True)
                        toggle_voice(3, True)
                        toggle_voice(4, True)
                else:
                    #print("base tone")
                    toggle_voice(2, False)
                    toggle_voice(3, False)
                    toggle_voice(4, False)
                
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
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


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point, handedness):
    
    #print(handedness.classification[0].label[0:])
    if len(landmark_point) > 0:
        for freq in chromatic_freqs:
            cv.circle(image, (int(freq), 240), 3, (0, 0, 0),
                    -1)

        centrepointx=(landmark_point[0][0]+landmark_point[4][0]+landmark_point[8][0]+landmark_point[12][0]+landmark_point[16][0]+landmark_point[20][0])/6
        centrepointy=(landmark_point[0][1]+landmark_point[4][1]+landmark_point[8][1]+landmark_point[12][1]+landmark_point[16][1]+landmark_point[20][1])/6

        cv.circle(image, (int(centrepointx), int(centrepointy)), 8, (100, 100, 255),
                    -1)
        cv.circle(image, (int(centrepointx), int(centrepointy)), 8, (0, 0, 0), 1)

        # Key Points
        
        cv.circle(image, tuple(landmark_point[0]), 8, (255, 255, 255),
                    -1)
        cv.circle(image, tuple(landmark_point[0]), 8, (0, 0, 0), 1)


        cv.circle(image, tuple(landmark_point[1]), 8, (100, 100, 100),
                    -1)
        cv.circle(image, tuple(landmark_point[1]), 8, (0, 0, 0), 1)

        #drawing thumb tip

        # 親指：指先
        cv.circle(image, tuple(landmark_point[4]), 8, (0, 255, 0),
                    -1)
        cv.circle(image, tuple(landmark_point[4]), 8, (0, 0, 0), 1)


        # index tip
        # 人差指：指先
        cv.circle(image, tuple(landmark_point[8]), 8, (0, 0, 255),
                    -1)
        cv.circle(image, tuple(landmark_point[8]), 8, (0, 0, 0), 1)                         

        # middle tip
        #中指：指先
        cv.circle(image, tuple(landmark_point[12]), 8, (255, 255, 0),
                    -1)
        cv.circle(image, tuple(landmark_point[12]), 8, (0, 0, 0), 1)


        # ring tip
        # 薬指：指先
        cv.circle(image, tuple(landmark_point[16]), 8, (255, 0, 0),
                    -1)
        cv.circle(image, tuple(landmark_point[16]), 8, (0, 0, 0), 1)
        

        # pinky tip
        # 小指：指先
        cv.circle(image, tuple(landmark_point[20]), 8, (0, 255, 255),
                    -1)
        cv.circle(image, tuple(landmark_point[20]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


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


if __name__ == '__main__':
    main()
