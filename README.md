# Shelf-Supervised Cross-Modal Pre-Training for 3D Object Detection



This repository contains the code for the paper **"Shelf-Supervised Cross-Modal Pre-Training for 3D Object Detection"**, presented at [CoRL 2024](https://www.corl.org/).

### [Paper (Arxiv)](https://arxiv.org/abs/2406.10115) | [Project Page](https://meharkhurana03.github.io/cm3d) | [BibTeX](#citation)

---

## Table of Contents

<!-- - [Introduction](#introduction) -->
- [Installation](#installation)
- [Generating Pseudo-Labels](#generating-pseudo-labels)
<!-- - [Dataset](#dataset) -->
<!-- - [Training](#training) -->
<!-- - [Evaluation](#evaluation) -->
<!-- - [Results](#results) -->
- [Citation](#citation)
<!-- - [Acknowledgements](#acknowledgements) -->

---

<!-- ## Introduction

State-of-the-art 3D object detectors typically require large annotated datasets. However, labeling 3D bounding boxes for LiDAR data is costly and time-consuming. Our approach leverages **shelf-supervised learning**, utilizing pre-trained image foundation models to generate pseudo-labels from multimodal data (RGB + LiDAR). The pseudo-labels enhance semi-supervised detection performance, especially when training with limited data.

Key contributions of the project:
- Cross-modal distillation from vision-language models to improve 3D object detection.
- Superior semi-supervised detection accuracy with **CM3D (Cross-Modal 3D Detection)**.
- Demonstrated improvements on benchmarks like nuScenes and Waymo Open Dataset (WOD).

--- -->

## Installation

To set up the repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/meharkhurana03/cm3d.git
   cd cm3d
   ```

2. Install the dependencies:
   ```bash
    conda create -n cm3d python=3.9 -f environment.yml
    conda activate cm3d
    ```

3. To run on nuScenes:
    ```bash
    pip install nuscenes-devkit
    ```

    ```bash
    cd src/nuscenes
    ```


## Generating Pseudo-Labels

First, generate 2D masks using detic:
```bash
python gen_2d_masks_trainset_detic.py
```

Next, generate pseudo-labels using the following command:
```bash
python 2d_to_3d_new.py
```

Note: Change the environment variables at the top of the scripts to point to the correct directories.


## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{khurana2024shelf,
        title={Shelf-Supervised Multi-Modal Pre-Training for 3D Object Detection},
        author={Khurana, Mehar and Peri, Neehar and Ramanan, Deva and Hays, James},
        journal={arXiv preprint arXiv:2406.10115},
        year={2024}
}
```