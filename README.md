# Kwai Project for Hand Pose Estimation

This is a project for hand pose estimation on top of _Simple Baselines for Human Pose Estimation and Tracking_, you can find more details in the [original paper](https://arxiv.org/pdf/1804.06208.pdf).

For more detailed information, please view this [sheet](https://docs.google.com/spreadsheets/d/1xdEpDyYqx8CaERFEAPpjGvlkzpMThUEdoCOGswbPmmI/edit?usp=sharing).

## Requirements

* Python == 3.5
* PyTorch >= 0.4.0
* torchvision >= 0.2.0
* OpenCV > 2.4.9

## Datasets

### Images

The current in-use and potentially to-be-used dataset is located at liujintao@gpu110:`/data/Kwai`.

Currently, the project is using:
* `CMU_Panoptic`: HandDB powered by CMU
* `RHD`: Rendered Handpose Dataset (in the wild) powered by Freiburg Univ.
* `union/real/kwai_*.jpg`: data provided by Kwai Inc., please keep it confidential

### Labels

The label files can be find at liujintao@gpu110:`/home/liujintao/app/Hand-Keypoint-Estimation/data/labels`.


## Usage

### Set up

Go to the root directory of the project, and follow the next steps:

* Compose the python environment using the `environment.yml`:
    ```bash
    conda create env -f environment.yml
    ```
* Set up environment variables for CUDA configuration:
    ```bash
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY}}
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export CUDA_HOME=/usr/local/cuda${CUDA_HOME:+:${CUDA_HOME}}
    ```  
    Note that this is **not required but recommended**.
*  Create a directory (`mkdir data`) for label files in the root directory of the project,
    and arrange the layout like this:
    ```
    └── labels
        ├── real
        │   ├── all_test_labels.txt
        │   ├── all_train_labels.txt
        │   ├── kwai_label_file.txt
        │   └── union_train_labels.txt
        └── synth
            ├── all_test_labels.txt
            ├── all_train_labels.txt
            ├── RHD_label.txt
            ├── union_train_labels.txt
            ├── vis_test_labels.txt
            └── vis_train_labels.txt
    ```
    The program will automatically generate a directory named `cache` and save pickle files under it in the first time you run the project 
    (it will take a couple of minutes).
    

### Train

```bash
python main.py --config config/train.yml --gpu 0
```

### Test
TBD : )