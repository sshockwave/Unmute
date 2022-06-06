import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm
from pathlib import Path
from preprocess import window_size

class CCTVDataset(torch.utils.data.Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        data_path = Path(data_path)
        self.data_path = data_path
        self.aud_data = np.load(data_path / 'audio.npy')
        self.cap = cv.VideoCapture((data_path / 'video.mp4').as_posix())
        num_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        assert num_frames == len(self.aud_data)
        breakpoints = np.load(data_path / 'breakpoints.npy')
        self.valid_frames = []
        last_start = 0
        for i in breakpoints:
            self.valid_frames += range(last_start, i - window_size + 1)
            last_start = i
        assert len(self.valid_frames) == num_frames - (window_size - 1) * len(breakpoints)

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, idx):
        frame_start = self.valid_frames[idx]
        self.cap.set(1, frame_start)
        vid = []
        for _ in range(window_size):
            ret, frame = self.cap.read()
            assert ret
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vid.append(frame_gray)
        vid = np.stack(vid, axis=0).astype(np.float32)
        margin = window_size // 2
        frame_mid = frame_start + margin
        return vid, self.aud_data[frame_mid].astype(np.float32)

def train_model(data_path, save_path, model=None):
    data_path, save_path = Path(data_path), Path(save_path)

    if model == None:
        from model import UnmuteTranscoder
        from preprocess_audio import NET_OUT
        model = UnmuteTranscoder(NET_OUT)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
        print('[INFO] CUDA is enabled')
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    from torch.utils.data import DataLoader
    train_dataset = CCTVDataset(data_path)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    mse = torch.nn.MSELoss()
    model.train()
    for epoch in range(10):
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            y_pred = model(x)
            loss = mse(y, y_pred)
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item(), epoch=epoch)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / 'chkpt.pt')

if __name__ == '__main__':
    train_model('./processed_data', './trained_weights')
