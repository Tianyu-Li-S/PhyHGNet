# PhyHGNet: Physically Guided Hypergraph Detection Network

This repository is the official implementation of the paper **"Physically Guided Hypergraph Detection Network for Micro-Defect Detection in Aerial Photovoltaic Thermal Imaging"**.

> **Note:** This project builds upon the excellent work of [RT-DETR](https://github.com/lyuwenyu/RT-DETR). We have deeply integrated physical prior modules (PLR-Block, TDP-Former, TBH Module, PAD-Conv) into the framework. Please refer to the original RT-DETR repository for the base architecture details.

## Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/Tianyu-Li-S/PhyHGNet.git
cd PhyHGNet

conda create -n phyhgnett python=3.8 -y
conda activate phyhgnett

# Install PyTorch (adjust the CUDA version according to your system)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

datasets/
└── pv_thermal/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json

python tools/train.py \
    -c configs/phyhgnett/phyhgnett_r18_6x_pv.yml \
    --use-amp \
    --batch-size 8 \
    --output-dir ./output/phyhgnett/

python tools/eval.py \
    -c configs/phyhgnett/phyhgnett_r18_6x_pv.yml \
    -r path/to/your/trained/model.pth \
    --batch-size 8

