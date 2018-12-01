# Kwai Project for Hand Pose Estimation

This is a pytroch version of Convolutional Pose Machines, which is adapted for hand pose estimation, you can find more details in the original paper [Convolutional Pose Machines](https://arxiv.org/pdf/1602.00134.pdf).

For more detailed information, please view this [sheet](https://docs.google.com/spreadsheets/d/1xdEpDyYqx8CaERFEAPpjGvlkzpMThUEdoCOGswbPmmI/edit?usp=sharing).

## Requirements

* Python == 3.5
* PyTorch >= 0.4.0
* torchvision >= 0.2.0
* OpenCV > 2.4.9

## The data set

The current in-use and potentially to-be-used dataset is located at liujintao@gpu105:/data/Kwai.

Currently, the project is using:
* `CMU_Panoptic`: HandDB powered by CMU
* `union/real/kwai_*.jpg`: data provided by Kwai Inc., please keep it confidential

Please copy at least `Kwai/CMU_Panoptic` and `Kwai/union` to your machine in advance.

BTW, we are going to use RHD dataset to facilitate training for source domain.
The images and label file can be found at `/data/Kwai/RHD/training/color` and `/data/Kwai/RHD/RHD_label.txt`.  


## Usage

### Set up

* Set the `PYTHONPATH` environment variable as below:
    ```bash
    export PYTHONPATH=/path/to/your/proj:/path/to/your/proj/dataset/:/path/to/your/proj/utils
    ```
* Compose the python environment using the `environment.yml`:
    ```bash
    conda create env -f environment.yml
    ```
* Since most of the servers in our lab have cuda>=9.1 preinstalled whereas the aforementioned `environment.yml`
only contains PyTorch compiled from cuda=9.0, that will be super slow if you use the original environment directly.
To address this issue, you may consider **one of** the following manner:
    + copy the directory located at `liujintao@gpu105:/usr/local/cuda-9.0` to your machine, and add the lines to your
    `~/.bashrc` or `~/.zshrc`
        ```bash
        export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY}}
        export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
        export CUDA_HOME=/usr/local/cuda-9.0${CUDA_HOME:+:${CUDA_HOME}}
        ```  
    + download PyTorch which is compiled over suited cuda version directly and this
    [site](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/) is recommended for speeding up.

### Train

* Set the train hyper-parameters in `config/train.yml`
* Set the available GPU devices in `scripts/cpm_train.sh`
* Run the command below:
    ```bash
    bash scripts/cpm_train.sh
    ```

### Test

Switch to the project root directory, and execute the command below:

```bash
CUDA_VISIBLE_DEVICES=0 python test/cpm_test.py --config ./config/test.yml
```

### Resume
**CAUTION**: this is a buggy part when you restore a model trained using multiple GPU devices
which I believe is a [intrinsic bug](https://github.com/roytseng-tw/Detectron.pytorch/issues/37) of PyTorch.
