from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import cv2 as cv

cascade_file = 'haarcascade_frontalface_alt.xml'
face_rows = 128
face_cols = 128
window_size = 9
channel_cnt = 1 # gray scale
pool_size = 2

def read_video(path: Path, face_cascade):
    # See https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    cap = cv.VideoCapture(path.as_posix())
    face_list = []
    pbar = tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
    pbar.set_description_str(path.as_posix())
    frame_id = -1
    first_warning = True
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
            minSize=(150,150),
        )
        if len(faces) != 1:
            if len(faces) == 0:
                tqdm.write(f'[ERROR] no faces found!')
                raise RuntimeError
            if first_warning:
                tqdm.write(f'[WARN] {len(faces)} faces detected in {path} (frame {frame_id}), using the largest')
                first_warning = False
        face = None
        max_w = -1
        for (x, y, w, h) in faces:
            if max_w < w:
                face = frame_gray[y:y+h, x:x+w]
                max_w = w
        face = cv.resize(face, (face_cols, face_rows))
        face_list.append(face)
    face_list = np.stack(face_list, axis=0)
    return face_list

def gen_data(vid, aud, vid_data_list=[], aud_data_list=[]):
    n = min(len(vid), len(aud)) # num frames
    margin = window_size // 2
    for i in range(margin, n - margin):
        vid_data_list.append(vid[i-margin: i+margin+1])
    aud_data_list.append(aud[margin:n-margin])
    return vid_data_list, aud_data_list

def process_videos(data_path, save_path):
    data_path, save_path = Path(data_path), Path(save_path)
    cascade_file_path = Path(__file__).parent / cascade_file
    face_cascade = cv.CascadeClassifier(cascade_file_path.as_posix())
    vid_data_list = []
    aud_data_list = []
    for f in tqdm(list(data_path.iterdir())):
        vid = read_video(f, face_cascade)
        from preprocess_audio import process_audio
        aud = process_audio(f)
        gen_data(vid, aud, vid_data_list, aud_data_list)
    vid_data_list = np.stack(vid_data_list, axis=0)
    aud_data_list = np.concatenate(aud_data_list, axis=0)
    assert len(vid_data_list) == len(aud_data_list)
    torch.save((vid_data_list, aud_data_list), save_path)

if __name__ == '__main__':
    process_videos('./data', './processed_data.pt')
