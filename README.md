# Unmute

[![State-of-the-art Shitcode](https://img.shields.io/static/v1?label=State-of-the-art&message=Shitcode&color=7B5804)](https://github.com/trekhleb/state-of-the-art-shitcode)

**Unmute** is a neural network model that can restore audio from the silenced video. Such a technique is similar to lip talk. At present, it is simple to generate video based on subtitles or sound, and some remarkable achievements have been made, such as DeepFake. However, it is much more difficult to generate audio from mute video.

## Dataset

We collected the videos of the famous CCTV channel News Broadcast (新闻联播) in 2022, and edited them to filter out the scenes of only the host. The total length of the videos is nearly 5 hours.

All videos are for learning and research purposes only.

* Data Source: https://tv.cctv.cn
* Dataset Download link: [Tsinghua Cloud Drive](https://cloud.tsinghua.edu.cn/d/7a563c890abb4e6eacb9/)

## Prerequisites

For the initial version:

```bash
conda install python=3.7.4 pytorch cudatoolkit=11.3 opencv tqdm scipy ffmpeg -c pytorch -c conda-forge
pip install -r requirements
```

## Acknowledgements
Our work is reproduction and optimization of Vid2speech: Speech Reconstruction from Silent Video (ICASSP 2017).
Also thanks to Learning Individual Speaking Styles for Accurate Lip to Speech Synthesis (CVPR 2020)!
