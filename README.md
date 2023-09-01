# [ICLR 2023] Stochastic Multi-Person 3D Motion Forecasting

<img src="demo/result.gif" width="100%"/>

## Introduction


This is the official implementation of "[Stochastic Multi-Person 3D Motion Forecasting](https://arxiv.org/abs/2306.05421)." 

## What's New
* [version 1.0] Code for diffusion-based DuMMF.

<!-- [[website](https://sirui-xu.github.io/STARS/)] [[arxiv](https://arxiv.org/abs/2302.04860)] [[demo](https://sirui-xu.github.io/STARS/images/demo.mp4)] [[poster](https://sirui-xu.github.io/assets/pdf/STARS_poster.pdf)] [[slides](https://sirui-xu.github.io/assets/pdf/STARS_slides.pdf)] [[talk](https://sirui-xu.github.io/STARS/images/talk.mp4)] -->
<!-- Our PF-Track illustrates significant advantages in:

* **Dramatically less ID-Switches**: PF-Track has **90\% less ID-Switches** compared to previous methods. So far, PF-Track is also SOTA in ID-Switches on nuScenes.
* **End-to-end perception and prediction**: PF-Track emulates an end-to-end framework.
* **Easy integration with detection heads**: PF-Track can cooperate with various DETR-style 3D detection heads.

Please **click** the gif below to check our full demo and reach out to [Ziqi Pang](https://ziqipang.github.io/) if you are interested. Our method seamlessly **address occlusions and hand-over between cameras**. -->
<!-- 
[![Demo video](./assets/video_demo_cover.gif)](https://www.youtube.com/watch?v=eJghONb2AGg) -->
## Citation
If you find our code or paper useful, please cite by:
```bibtex
@inproceedings{
xu2023stochastic,
title={Stochastic Multi-Person 3D Motion Forecasting},
author={Sirui Xu and Yu-Xiong Wang and Liangyan Gui},
booktitle={ICLR},
year={2023},
}
```

<!-- ## Getting Started

Please follow our documentation step by step. For the convenience of developers and researchers, we also add notes for developers to better convey the implementations of PF-Track and accelerate your adaptation of our framework. If you like my documentation and help, please recommend our work to your colleagues and friends.

1. [**Pretrained models and data files.**](./documents/pretrained.md)
2. [**Environment Setup.**](./documents/environment.md)
3. [**Preprocessing nuScenes.**](./documents/preprocessing.md)
4. [**Training.**](./documents/training.md)
5. [**Inference.**](./documents/inference.md)

## Guide for Developers and Researchers

It literally took us **THREE MONTHS** to implement the baseline because designing the end-to-end tracking and prediction framework is challenging. Therefore, we write the following documents to help you better understand our design choices, read the code, and adapt them to your own tasks and datasets.

1. [**System Overview: An ABC Guide to End-to-end MOT.**](./documents/designs.md) (Please skim through it even if you know end-to-end MOT well, because we clarify several implementation details that are non-trivial.)
2. [**Visualization tools.**](./documents/visualization.md)
3. [**Integration with various detection heads.**](./documents/detection_heads.md)

## Acknowledgements

We thank the contributors to the following open-source projects. Our project is impossible without the inspirations from these excellent researchers and engineers.

* 3D Detection. [MMDetection3d](https://github.com/open-mmlab/mmdetection3d), [DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR).
* Multi-object tracking. [MOTR](https://github.com/megvii-research/MOTR), [MUTR3D](https://github.com/a1600012888/MUTR3D), [SimpleTrack](https://github.com/tusen-ai/SimpleTrack).
* End-to-end motion forecasting. [FutureDet](https://github.com/neeharperi/FutureDet).

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. -->