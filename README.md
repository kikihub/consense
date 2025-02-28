# ConSense: Continually Sensing Human Activity with WiFi via Growing and Picking

[![arXiv](https://img.shields.io/badge/arXiv-2502.17483-b31b1b.svg)](https://arxiv.org/abs/2502.17483)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📁 Data Processing

### Data Source
Download them to consense/data/raw_data/xxx
- WiAR：https://github.com/linteresa/WiAR
- MMFi：https://github.com/ybhbingo/MMFi_dataset
- XRF：https://github.com/aiotgroup/XRF55-repo

### Preprocessing Process
step1：
   ```
   python consense/data/preprocess/xxx/process_step1.py
   ```
step2：
   ```
   python consense/data/preprocess/xxx/process_step2.py
   ```

### Data Directory
```
data/
├── preprocess/         # Preprocessing steps
├── raw_data/           # Raw data
└── processed_data/     # Processed data
```

## 🚀 Training

WiAR Short Task
```
python main.py --dataset=wiar --n_tasks=8 --n_warmup_epochs=20 --n_epochs=50 --half_iid=1
```
WiAR Long Task
```
python main.py --dataset=wiar --n_tasks=8 --n_warmup_epochs=20 --n_epochs=50 --half_iid=0
```
MMFi Short Task
```
python main.py --dataset=mmfi --n_tasks=9 --n_warmup_epochs=20 --n_epochs=50 --half_iid=1
```
MMFi Long Task
```
python main.py --dataset=mmfi --n_tasks=9 --n_warmup_epochs=20 --n_epochs=50 --half_iid=0
```
XRF Short Task
```
python main.py --dataset=xrf --n_tasks=8 --n_warmup_epochs=20 --n_epochs=50 --half_iid=1
```
XRF Long Task
```
python main.py --dataset=xrf --n_tasks=8 --n_warmup_epochs=20 --n_epochs=50 --half_iid=0
```

## 📚 Citations
If you find our works useful in your research, please consider citing:
```bibtex
@inproceedings{rong2025consense,
  title={{ConSense: Continually Sensing Human Activity with WiFi via Growing and Picking}},
  author={Li, Rong and Deng, Tao and Feng, Siwei and  Sun, Mingjie and Jia, Juncheng},
  booktitle={Proceedings of the AAAI Conference on Artificial intelligence},
  pages={1--8},
  year={2025}
}
```