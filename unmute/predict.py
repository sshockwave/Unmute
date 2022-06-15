import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import audio_tools as aud
from preprocess import window_size
from preprocess_audio import LPC_ORDER, SPF

class SingleVideoDataset(torch.utils.data.Dataset):
    def __init__(self, vid_path) -> None:
        super().__init__()
        from preprocess import read_video, get_face_cascade
        self.data = read_video(vid_path, get_face_cascade())

    def __len__(self):
        return len(self.data) - window_size + 1

    def __getitem__(self, idx):
        return self.data[idx:idx + window_size].astype(np.float32)

def get_lsf(Y_pred):
    lsf_tepr = Y_pred[:,:-2]
    assert lsf_tepr.shape[1] % 2 == 0
    lsf_tepr2 = np.zeros((lsf_tepr.shape[0]*2,lsf_tepr.shape[1] // 2))
    lsf_tepr2[::2,:] = lsf_tepr[:,:LPC_ORDER]
    lsf_tepr2[1::2,:] = lsf_tepr[:,LPC_ORDER:]
    g_tepr = Y_pred[:,-2:]
    g_tepr2 = np.zeros((g_tepr.shape[0]*2,1))
    g_tepr2[::2,:] = g_tepr[:,:1]
    g_tepr2[1::2,:] = g_tepr[:,1:]
    g_tepr2[g_tepr2<0] = 0.0
    return lsf_tepr2, g_tepr2

def y_to_wav(y, output_file, pad_len=(window_size // 2) * SPF, excite=None):
    r"""
    y: np.ndarray
    """
    lsf, g = get_lsf(y)
    lpc = aud.lsf_to_lpc(lsf)
    assert SPF % 2 == 0
    x = aud.lpc_synthesis(lpc, g, excite, window_step=SPF//2)
    pad = np.zeros([pad_len])
    x = np.hstack((np.hstack((pad,x)),pad))
    from scipy.io.wavfile import write
    from preprocess_audio import SR
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write(output_file, SR, x)

def predict_audio(input_file, model_path):
    from model import UnmuteTranscoder
    from preprocess_audio import NET_OUT
    model = UnmuteTranscoder(NET_OUT)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
        print('[INFO] CUDA is enabled')
    model = model.to(device)
    pred_dataset = SingleVideoDataset(Path(input_file))
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=64, shuffle=False)
    y_pred = []
    with torch.no_grad():
        model = model.to(device)
        for vid in tqdm(pred_loader):
            y_pred.append(model(vid.to(device)))
        y_pred = torch.cat(y_pred, dim=0)
    return y_pred.cpu().numpy()

def main():
    from preprocess_audio import process_audio
    y_ans, e = process_audio('./data/0.mp4')
    y_to_wav(y_ans, './output/0_original.wav', pad_len=0, excite=e)
    y_to_wav(y_ans, './output/0_original_no_excite.wav', pad_len=0)
    y_pred = predict_audio('./data/0.mp4', './trained_weights/chkpt.pt')
    y_to_wav(y_pred, './output/0_predict.wav', pad_len=0)
    pass

if __name__ == "__main__":
    main()
