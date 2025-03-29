
<p align="center">
    <h1 align="center">
        <img src="https://sam2act.github.io/static/images/img_logo.png" width="25px"/>
        SAM2Act:
        <br>
        Integrating Visual Foundation Model with 
        <br>
        A Memory Architecture for Robotic Manipulation
    </h1>
</p>

<p align="center">
  <a href="https://hq-fang.github.io">Haoquan Fang</a>, 
  <a href="https://www.markusgrotz.com">Markus Grotz</a>, 
  <a href="https://wpumacay.github.io">Wilbert Pumacay</a>, 
  <a href="https://helen9975.github.io">Yi Ru Wang</a>, 
  <br>
  <a href="https://homes.cs.washington.edu/~fox">Dieter Fox<sup>†</sup></a>, 
  <a href="https://ranjaykrishna.com">Ranjay Krishna<sup>†</sup></a>, 
  <a href="https://duanjiafei.com">Jiafei Duan<sup>†</sup></a>
  <br>
  <sup>†</sup>Equal Advising
</p>

<div align="center">
  <p>
    <a href="https://sam2act.github.io/">
      <img src="https://img.shields.io/badge/Website-grey?logo=google-chrome&logoColor=white&labelColor=blue">
    </a>
    <a href="https://arxiv.org/abs/2501.18564">
      <img src="https://img.shields.io/badge/arXiv-grey?logo=arxiv&logoColor=white&labelColor=red">
    </a>
    <a href="https://huggingface.co/datasets/hqfang/MemoryBench">
      <img src="https://img.shields.io/badge/MemoryBench-grey?logo=huggingface&logoColor=white&labelColor=yellow">
    </a>
    <a href="https://x.com/DJiafei/status/1884954101697699940">
      <img src="https://img.shields.io/badge/Post-grey?logo=x&logoColor=white&labelColor=black">
    </a>
  </p>
  <p>
    <a href="https://paperswithcode.com/sota/robot-manipulation-on-rlbench?p=sam2act-integrating-visual-foundation-model-1">
      <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sam2act-integrating-visual-foundation-model-1/robot-manipulation-on-rlbench">
    </a>
  </p>
</div>

<br>

<p align="center">
  <a href="https://sam2act.github.io/static/videos/vid_intro.mp4">
    <img src="./vid_intro.gif" alt="Watch the video" width="600">
  </a>
</p>


SAM2Act is a multi-view robotics transformer-based policy that enhances feature representation by integrating multi-resolution upsampling with visual embeddings from large-scale foundation model. Built on the RVT-2 multi-view transformer, SAM2Act achieves strong multitask success and generalization. Building on this foundation, we introduce SAM2Act+, which incorporates a memory-based architecture inspired by SAM2's approach. Using a memory bank, an encoder, and an attention mechanism, SAM2Act+ enables episodic recall to solve spatial memory-dependent manipulation tasks.

---

This is the official repository of [SAM2Act](https://sam2act.github.io). If you find our work useful, please consider citing our paper:
```
@misc{fang2025sam2act,
      title={SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation}, 
      author={Haoquan Fang and Markus Grotz and Wilbert Pumacay and Yi Ru Wang and Dieter Fox and Ranjay Krishna and Jiafei Duan},
      year={2025},
      eprint={2501.18564},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2501.18564}, 
}
```

## Table of Contents

- [Environment Setup](#environment-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Common Questions](#common-questions)
- [Acknowledgement](#acknowledgement)

## Environment Setup

### Install
- Tested (Recommended) Versions: Python 3.10. We used CUDA 11.8. 

- **Step 1 (Optional):**
We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) and creating a virtual environment.
```
conda create --name sam2act python=3.10
conda activate sam2act
```

- **Step 2:** Install PyTorch. Make sure the PyTorch version is compatible with the CUDA version. One recommended version compatible with CUDA 11.8 and PyTorch3D can be installed with the following command. More instructions to install PyTorch can be found [here](https://pytorch.org/).
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- **Step 3:** Install PyTorch3D.

The original RVT repository recommends that you can skip this step if you just want to use RVT-2 backbone and its custom Point-Renderer for rendering. If you want to try out RVT backbone or different renderer, PyTorch3D is required. However, we still recommend installing this as there are some nested dependencies that requires PyTorch3D package.

One recommended version that is compatible with the rest of the library can be installed as follows. Note that this might take some time. For more instructions visit [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
```
FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

- **Step 4:** Install CoppeliaSim. PyRep requires version **4.1** of CoppeliaSim. Download and unzip CoppeliaSim: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, add the following to your *~/.bashrc* file. (__NOTE__: the 'EDIT ME' in the first line)

```
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
```
Remember to source your .bashrc (`source ~/.bashrc`) or  .zshrc (`source ~/.zshrc`) after this.

Note that when you are using a headless device, it seems that the last line is unnecessary, and sometimes even cause problems.

- **Step 5:** Clone this repository and install sam2act and other submodules.

To locally install the repository, you can either `pip install -e '.[xformers]'` to install the library with [xformers](https://github.com/facebookresearch/xformers) or `pip install -e .` to install without it. We recommend using the former as improves speed. However, sometimes the installation might fail due to the xformers dependency. In that case, you can install the library without xformers. The performance difference between the two is minimal but speed could be slower without xformers.
```
pip install -e '.[xformers]' 
```
Note that for bug-free implementation, we still suggest installing without `xformers` as below.
```
pip install -e .
```
Install, required libraries for PyRep, RLBench, YARR, PerAct Colab, and Point Renderer.
```
pip install -e sam2act/libs/PyRep 
pip install -e sam2act/libs/RLBench 
pip install -e sam2act/libs/YARR 
pip install -e sam2act/libs/peract_colab
pip install -e sam2act/libs/point-renderer
``` 

You may also want to upgrade some packages if there is any error:
```
pip install --upgrade hydra-core
``` 
 
- **Step 6:** Download SAM2 weights and dataset.
    - Before starting, download SAM2 pretrained weights for loading SAM2Act using the following command.
    ```
    cd sam2act/mvt/sam2_train/checkpoints
    download_ckpts.sh
    ``` 

    - For experiments on RLBench, we use [pre-generated dataset](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) provided by [PerAct](https://github.com/peract/peract#download). If downloading from Google Drive encounters any limits, we also provide a mirror of the same dataset in [Hugging Face](https://huggingface.co/datasets/hqfang/RLBench-18-Tasks). Please download and place them under `SAM2Act/sam2act/data/xxx` where `xxx` is either `train`, `test`, or `val`.  

    - Additionally, building upon PerAct's dataloader, we create a new dataloader that can sample observation sequences of a given size, and it also supports the same functionality as PerAct's. Same as PerAct, our dataloader is also based on [YARR](https://github.com/stepjam/YARR). YARR creates a replay buffer on the fly which can increase the startup time. We provide an option to directly load the replay buffer from the disk. We recommend using the pre-generated replay buffer (98 GB) as it reduces the startup time. You can download replay buffer for [indidual tasks](https://huggingface.co/datasets/hqfang/SAM2Act/tree/main/replay_temporal/replay_train). After downloading, uncompress the replay buffer(s) (for example using the command `tar -xf <task_name>.tar.xz`) and place it under `SAM2Act/sam2act/replay_temporal/replay_xxx` where `xxx` is `train` (for now we only provide replay buffer for trianing split). Note that is useful only if you want to train SAM2Act from scratch and not needed if you want to evaluate the pre-trained model.

    - If you prefer using dataloader same as PerAct's, you can refer to the step 6 of this [instruction](https://github.com/NVlabs/RVT?tab=readme-ov-file#getting-started). You also need to change `get_dataset_temporal` to `get_dataset` in `train.py`. Again, note that this is not necessary because our dataloader preserves all functionality of PerAct's.

    - For experiments on MemoryBench, we also provide a [pre-generated dataset](https://huggingface.co/datasets/hqfang/MemoryBench). Please download and place them under `SAM2Act/sam2act/data_memory/xxx` where `xxx` is either `train` or `test`.  


## Training 
### Training SAM2Act on RLBench

To train SAM2Act on all RLBench tasks, use the following command (from folder `SAM2Act/sam2act`):
```
WANDB_MODE="offline" \
torchrun --nproc_per_node="8" --nnodes="1" \
  train.py \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml
```
In this example, we use 8 H100 GPUs on 1 node. Change the `nproc_per_node` and `nnodes` flags depending on available compute. Note that `bs` in config denotes batch size per GPU, and `lr` will be adjusted by total batch size by default following previous works. Be careful when using different number of GPUs.

### Training SAM2Act on MemoryBench
By default, the training code is using RLBench data. To train SAM2Act on MemoryBench, change the argument for function `get_dataset_temporal` in `train.py`. Following the instruction in code, comment the directory variables for RLBench, and uncomment the ones for MemoryBench. Then, similarly, use the following command (from folder `SAM2Act/sam2act`):
```
WANDB_MODE="offline" \
torchrun --nproc_per_node="8" --nnodes="1" \
  train.py \
  --exp_cfg_opts "tasks <name_of_memorybench_task>" \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml
```
This overrides the name of the task in `configs/sam2act.yaml`, allowing the code run with the expected task in MemoryBench. For more instruction on how to override config, see more below. You can also directly change the variables in `configs/sam2act.yaml`.

### Training SAM2Act+ on MemoryBench
Make sure that Stage 1 training is done by following previous instruction. Then, to train SAM2Act+ on MemoryBench, use the following command (from folder `SAM2Act/sam2act`):
```
WANDB_MODE="offline" \
torchrun --nproc_per_node="8" --nnodes="1" \
  train_plus.py \
  --exp_cfg_opts "tasks <memorybench_task_name>" \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml
```
By default, we use task `put_block_back` in `configs/sam2act_plus.yaml`. You can override this with any task in MemoryBench. Make sure that the `configs/sam2act_plus.yaml` has the same `task_id` with `configs/sam2act.yaml`, because the second stage training requires finding previous pre-trained weight in the same folder. Note that the only two differences between `mvt/configs/sam2act.yaml` and `mvt/configs/sam2act_plus.yaml` are that in `mvt/configs/sam2act_plus.yaml`, `use_memory` is set to be `True` and `num_maskmem` is valid during training. Make sure that `bs` in `configs/sam2act_plus.yaml` equals to `num_maskmem + 1`.

### More details about `train.py` and `train_plus.py`
- wandb in offline mode is used by default, if you want to attach your wandb api key, please change the first line of training command and use the following command:
```
WANDB_API_KEY="<your_wandb_api_key>" \
torchrun --nproc_per_node="8" --nnodes="1" \
  train.py \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml
```
  - if you want to turn wandb off, you can either change `wandb` in `configs/sam2act.yaml` and `configs/sam2act_plus.yaml` to `False`, or override it as:
  ```
  torchrun --nproc_per_node="8" --nnodes="1" \
    train.py \
    --exp_cfg_opts "wandb False" \
    --exp_cfg_path configs/sam2act.yaml \
    --mvt_cfg_path mvt/configs/sam2act.yaml
  ```
- default parameters for an `experiment` are defined [here](https://github.com/sam2act/SAM2Act/blob/master/sam2act/config.py).
- default parameters for `rvt` are defined [here](https://github.com/sam2act/SAM2Act/blob/master/sam2act/mvt/config.py).
- the parameters in for `experiment` and `rvt` can be overwritten by two ways:
    - specifying the path of a yaml file
    - manually overwriting using a `opts` string of format `<param1> <val1> <param2> <val2> ..`
- Manual overwriting has higher precedence over the yaml file.

```
WANDB_MODE="offline" \
torchrun --nproc_per_node="8" --nnodes="1" \
  train.py \
  --exp_cfg_opts <> \
  --mvt_cfg_opts <> \
  --exp_cfg_path <> \
  --mvt_cfg_path <>
```

The following command overwrites the parameters for the `experiment` with the `configs/sam2act.yaml` file. It also overwrites the `bs` parameters through the command line.
```
WANDB_MODE="offline" \
torchrun --nproc_per_node="8" --nnodes="1" \
  train.py \
  --exp_cfg_opts "bs 4" \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml
```

## Evaluation
### Evaluate SAM2Act on RLBench
Download the [pretrained SAM2Act model](https://huggingface.co/datasets/hqfang/SAM2Act/tree/main/sam2act_rlbench). Place the model (`model_89.pth` trained for 90 epochs or 56.25K steps with batch size 256 using 32 A100 GPUs) and the config files under the folder `SAM2Act/sam2act/runs/sam2act_rlbench/`. The model checkpoint excludes optimizer state to save disk space. Run evaluation using (from folder `SAM2Act/sam2act`):
```
python eval.py \
  --model-folder runs/sam2act_rlbench \
  --eval-datafolder ./data/test \
  --tasks all \
  --eval-episodes 25 \
  --log-name test/1 \
  --device 0 \
  --headless \
  --model-name model_89.pth
```

Note that the training process involves significant randomness, primarily due to how data is sampled in each batch (retraining may get different results as well). Additionally, randomness is introduced during evaluation by the sampling-based motion planner used in RLBench. As a result, the evaluation results of the pretrained SAM2Act may not be perfectly aligned with those reported in the paper, but they remain nearly identical. We evaluated the newly trained model four times, obtaining an average success rate of 86.8 ± 1.1.

### Evaluate SAM2Act+ on MemoryBench
Download the [pretrained SAM2Act+ model (coming soon)]() for each task. Place the model (`model_plus_19.pth` trained for 20 epochs or 12.5K steps with batch size 320) and the config files under the folder `SAM2Act/sam2act/runs/sam2act_plus_<task_name>/`. Run evaluation using (from folder `SAM2Act/sam2act`):
```
python eval.py \
  --model-folder runs/sam2act_plus_<task_name> \
  --eval-datafolder ./data_memory/test \
  --tasks <memorybench_task_name> \
  --eval-episodes 25 \
  --log-name test/1 \
  --device 0 \
  --headless \
  --model-name model_plus_19.pth
```

## Common Questions
We are running our code mainly on a headless server. Here are some solutions to the issues we met.
- If you get error like `cannot find file libcoppeliaSim.so.1`, this might because that library specifically looks for `libcoppeliaSim.so.1`, but the actual shared library file is named `libcoppeliaSim.so`. Try running:
```
ln -sf /path/CoppeliaSim/libcoppeliaSim.so /path/CoppeliaSim/libcoppeliaSim.so.1
```
- If you get error saying no display found when running evaluation, make sure you have added the `headless` flag, then try to install `xvfb`, and run the eval command as something like:
```
xvfb-run -a -s "-screen 0 1400x900x24" \
  python eval.py \
  ...
```
Below are some common questions when running RVT's repository. Since our repository is built upon it, those problems might also be valuable.
- If you face issues installing `xformers` and PyTorch3D, information in this issue might be useful https://github.com/NVlabs/RVT/issues/45.

- If you get qt plugin error like `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" <somepath>/cv2/qt/plugins" even though it was found`, try uninstalling opencv-python and installing opencv-python-headless

```
pip uninstall opencv-python
pip install opencv-python-headless
```

- If you are having issues running evaluation on a headless server, please refer to https://github.com/NVlabs/RVT/issues/2#issuecomment-1620704943.

- If you want to generate visualization videos, please refer to https://github.com/NVlabs/RVT/issues/5.

If these still cannot solve your issue, please try search on RVT's [issues](https://github.com/NVlabs/RVT/issues). If the problem still persists after that, please feel free to raise an issue to this repository.
## Acknowledgement
We sincerely thank the authors of the following repositories for sharing their code.

- [PerAct](https://github.com/peract/peract)
- [PerAct Colab](https://github.com/peract/peract_colab/tree/master)
- [PyRep](https://github.com/stepjam/PyRep)
- [RLBench](https://github.com/stepjam/RLBench/tree/master)
- [YARR](https://github.com/stepjam/YARR)
- [RVT](https://github.com/NVlabs/RVT)
- [SAM-E](https://github.com/pipixiaqishi1/SAM-E)
- [SAM2](https://github.com/facebookresearch/sam2)
- [The COLOSSEUM](https://robot-colosseum.github.io/)



