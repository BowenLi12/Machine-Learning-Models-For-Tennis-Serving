import os
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from yolo_classes import YOLOClasses

# global path for images
IMAGES_DIR = "/media/user/Storage/Documents/Coding_Minds/tennis_height/tennis_backend_python/images"
os.makedirs(IMAGES_DIR, exist_ok=True)  # Ensure the directory exists

yolo_model = YOLO('models/yolov8n.pt')
path = 'models/ball_model.pt'
ball_model = YOLO(path)

# Paths for directories
GOOD_VIDEOS_DIR = "/media/user/Storage/Documents/Coding_Minds/tennis_height/tennis_backend_python/data/tennis good/tennis good"
BAD_VIDEOS_DIR = "/media/user/Storage/Documents/Coding_Minds/tennis_height/tennis_backend_python/data/tennis bad/tennis bad"
OUTPUT_CSV_DIR = "/media/user/Storage/Documents/Coding_Minds/tennis_height/tennis_backend_python/data/output_csv_files"
MASTER_CSV_PATH = "/media/user/Storage/Documents/Coding_Minds/tennis_height/tennis_backend_python/data/master_dataset.csv"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)  # Ensure the output directory exists

yolo_model = YOLO('models/yolov8n.pt')
path = 'models/ball_model.pt'
ball_model = YOLO(path)


def make_directory(name: str):
    if not os.path.isdir(name):
        os.mkdir(name)


def resize_image(image):
    h, w, _ = image.shape
    h, w = h // 2, w // 2
    image = cv2.resize(image, (w, h))
    return image, h, w


def detect_person_ball_racket(frame):
    results = yolo_model(frame, verbose=False)
    person, racket = None, None
    annotator = Annotator(frame)
    r = results[0]
    min_person_size = 5000

    boxes = r.boxes
    for box in boxes:
        c = box.cls
        b = box.xyxy[0]

        if int(c) == YOLOClasses.person:
            box_size = (b[2] - b[0]) * (b[3] - b[1])
            if box_size > min_person_size:
                min_person_size = box_size
                person = b
        elif int(c) == YOLOClasses.tennis_racket:
            racket = b

    return person, racket, annotator.result()


def detect_ball(frame):
    results = ball_model(frame, verbose=False)
    ball = None
    annotator = Annotator(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            ball = b

    return ball, annotator.result()


def calculate_distance(ball_pos, person_pos):
    ball_x, ball_y = (ball_pos[0] + ball_pos[2]) / 2, (ball_pos[1] + ball_pos[3]) / 2
    person_x, person_y = (person_pos[0] + person_pos[2]) / 2, (person_pos[1] + person_pos[3]) / 2
    return np.sqrt((ball_x - person_x) ** 2 + (ball_y - person_y) ** 2)


def process_video(video_path, category):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame_time = round(frame_no / fps, 1)
        frame, h, w = resize_image(frame)
        ball_pos, frame = detect_ball(frame)
        person_pos, racket_pos, frame = detect_person_ball_racket(frame)

        if ball_pos is not None and racket_pos is not None:
            distance = calculate_distance(ball_pos, racket_pos)
            frames.append(
                {'frame_time': frame_time, 'distance': distance, 'ball_pos': ball_pos,
                 'racket_pos': racket_pos, 'person_pos': person_pos, 'category': category}
            )

    cap.release()
    return frames, h, w


def select_frames(frames, h, w):
    max_distance = 110
    result = []

    for frame_info in frames:
        if frame_info['distance'] <= max_distance:
            ball_center = get_center(frame_info['ball_pos'], h, w)
            racket_center = get_center(frame_info['racket_pos'], h, w)
            person_center = get_center(frame_info['person_pos'], h, w)
            result.append((frame_info['frame_time'], ball_center[0], ball_center[1],
                           racket_center[0], racket_center[1], person_center[0], person_center[1],
                           frame_info['category']))

    return result


def get_center(object_pos, height, width):
    object_x = ((object_pos[0] + object_pos[2]) / 2).item() / width
    object_y = ((object_pos[1] + object_pos[3]) / 2).item() / height
    return round(object_x, 2), round(object_y, 2)


def convert_to_dataframe(result):
    return pd.DataFrame(result, columns=['frame_time', 'ball_center_x', 'ball_center_y',
                                         'racket_center_x', 'racket_center_y',
                                         'person_center_x', 'person_center_y', 'target'])


def save_csv(df, csv_path):
    df.to_csv(csv_path, index=False)


def process_videos_in_directory(video_dir, category):
    master_data = []
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".MOV"):
            video_path = os.path.join(video_dir, video_file)
            frames, h, w = process_video(video_path, category)
            if frames:
                selected_frames = select_frames(frames, h, w)
                video_df = convert_to_dataframe(selected_frames)

                csv_filename = f"{video_file.split('.')[0]}.csv"
                csv_path = os.path.join(OUTPUT_CSV_DIR, csv_filename)
                save_csv(video_df, csv_path)

                master_data.extend(selected_frames)
                print(f"Processed: {video_file}")

    return master_data


if __name__ == '__main__':
    master_data = []

    print("Processing good form videos...")
    master_data.extend(process_videos_in_directory(GOOD_VIDEOS_DIR, 1))

    print("Processing bad form videos...")
    master_data.extend(process_videos_in_directory(BAD_VIDEOS_DIR, 0))

    master_df = convert_to_dataframe(master_data)
    save_csv(master_df, MASTER_CSV_PATH)
    print(f"Master CSV saved at {MASTER_CSV_PATH}")
