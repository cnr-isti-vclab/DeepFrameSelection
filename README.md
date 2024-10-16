Deep Frame Selection
====================
Deep Frame Selection (DFS) is a deep-learning based metric for selecting frames from a video for Structure-from-Motion (SfM) pipelines or 3D reconstruction. The metric selects high-quality frames without repetitions and low quality frames.

![HDR-VDP](images/pipeline.jpg?raw=true "Our pipeline for selecting frames.")


DEPENDENCIES:
==============

Requires the PyTorch library along with Pillow, NumPy, tqdm, torchvision, Matplotlib, glob2, pandas, and scikit-learn.

As the first step, you need to follow the [instructions for installing PyTorch](http://pytorch.org/).

To install dependencies, please use the following command: 

```bash
pip3 install numpy, matplotlib, tqdm, glob2, torchvision, pandas, pillow, scikit-learn, opencv-python. 
```

HOW TO RUN IT:
==============
To run our metric on a video (MP4) or a folder of images (e.g., PNGs or JPEGs), you need to launch the file ```dfs.py```. Some examples:

Testing DFS on videos:

```
python3 dfs.py video.mp4
```

WEIGHTS DOWNLOAD:
=================
Weights are included in this repository.

DATASET PREPARATION:
====================
Coming soon.

TRAINING:
=========
Coming soon.


REFERENCE:
==========

If you use DFS in your work, please cite it using this reference:

```
@INPROCEEDINGS{9506227,
  author={Banterle, Francesco and Gong, Rui and Corsini, Massimiliano and Ganovelli, Fabio and Gool, Luc Van and Cignoni, Paolo},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={A Deep Learning Method for Frame Selection in Videos for Structure from Motion Pipelines}, 
  year={2021},
  volume={},
  number={},
  pages={3667-3671},
  keywords={Time-frequency analysis;Structure from motion;Video sequences;Pipelines;Computer architecture;Streaming media;Prediction algorithms;Structure from Motion;Deep Learning;Point-Cloud Generation;Video Processing},
  doi={10.1109/ICIP42928.2021.9506227}}
```
