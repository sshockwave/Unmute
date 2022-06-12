# Unmute

[![State-of-the-art Shitcode](https://img.shields.io/static/v1?label=State-of-the-art&message=Shitcode&color=7B5804)](https://github.com/trekhleb/state-of-the-art-shitcode)

**Unmute** is a neural network model that can restore audio from the silenced video. Such a technique is similar to lip talk. At present, it is simple to generate video based on subtitles or sound, and some remarkable achievements have been made, such as DeepFake. However, it is much more difficult to generate audio from mute video.

Our work is reproduction and optimization of Vid2speech: Speech Reconstruction from Silent Video (ICASSP 2017). 

We collected the videos of the famous CCTV channel News Broadcast (新闻联播) in 2022, and edited them to filter out the scenes of only the host. The total length of the videos is nearly 5 hours.

All videos are for learning and research purposes only.

Data Source: https://tv.cctv.cn

## Prerequisites

```bash
conda install pytorch cudatoolkit=11.3 opencv tqdm scipy -c pytorch -c conda-forge
```
