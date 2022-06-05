from pathlib import Path
import torch
from tqdm import tqdm
import cv2 as cv

cascade_file = 'haarcascade_frontalface_alt.xml'
face_rows = 128
face_cols = 128
window_size = 9
channel_cnt = 1 # gray scale
pool_size = 2

def read_video(path: Path, face_cascade, data_list):
    # See https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    cap = cv.VideoCapture(path.as_posix())
    face_list = []
    pbar = tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
    frame_id = -1
    while True:
        frame_id += 1
        ret, frame = cap.read()
        if not ret:
            # File ended
            break
        pbar.update()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            frame_gray,
            minSize=(70,70),
        )
        if len(faces) != 1:
            tqdm.write(f'[ERROR] {len(faces)} faces detected in {path} (frame {frame_id})')
        for (x, y, w, h) in faces:
            face = frame_gray[y:y+h, x:x+w]
        face = cv.resize(face, (face_cols, face_rows))
        face_list.append(face)
    margin = window_size // 2
    for i in range(margin, len(face_list) - margin):
        data_list.append(torch.tensor(face_list[i-margin: i+margin]))

def process_videos(data_path, save_path):
    data_path, save_path = Path(data_path), Path(save_path)
    cascade_file_path = Path(__file__).parent / cascade_file
    face_cascade = cv.CascadeClassifier(cascade_file_path.as_posix())
    data_list = []
    for f in tqdm(list(data_path.iterdir())):
        read_video(f, face_cascade, data_list)
    return data_list
    r"""
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, 24, (face_cols, face_rows))
    for frame in data_list:
        out.write(frame)
    """
