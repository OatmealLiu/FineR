<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"><img src="figures/logo.png" width="256"></h1>
  <h1 align="center">Democratizing Fine-grained Visual Recognition with Large Language Models</h1>
  <p align="center">
    <a href="https://oatmealliu.github.io/"><strong>Mingxuan Liu</strong></a>
    ·
    <a href="https://roysubhankar.github.io/"><strong>Subhankar Roy</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=uBjSytAAAAAJ&hl=en"><strong>Wenjing Li</strong></a>
    ·
    <a href="https://zhunzhong.site/"><strong>Zhun Zhong</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en"><strong>Nicu Sebe</strong></a>
    ·
    <a href="https://scholar.google.ca/citations?user=xf1T870AAAAJ&hl=en"><strong>Elisa Ricci</strong></a>
  </p>
  <h2 align="center">ICLR 2024</h2>
  <h3 align="center"><a href="https://openreview.net/forum?id=c7DND1iIgb">OpenReview</a> | <a href="https://arxiv.org/abs/2401.13837">Paper</a> | <a href="https://projfiner.github.io/">Project Page</a></h3>
<div align="center"></div>
<p align="center">
  <p>
  <strong>TL;DR</strong>: We propose <strong>Fine-grained Semantic Category Reasoning (FineR)</strong> system to address fine-grained visual recognition without needing expert annotations and knowing category names as a-priori. FineR leverages large language models to identify fine-grained image categories by interpreting visual attributes as text. This allows it to reason about subtle differences between species or objects, outperforming current FGVR methods.
  </p>
  <a href="">
    <img src="figures/teaser_finer.png" alt="Logo" width="100%">
  </a>
<br>


## 📣 News:
- [02/29/2024] We released the code along with the intermediate results (in `experiments/`, including: super-class, attributes, attribute-description pairs, LLM-prompts, LLM raw answers, parsed LLM answers). User instructions coming later.
- [01/15/2024] Our work is accepted to <a href="https://iclr.cc/Conferences/2024"><strong>ICLR 2024</strong></a> 🌼! Code is coming soon. See you in Vienna this May!


## 💾 Installation
Requirements:
- Linux or macOS with Python ≥ 3.9
- PyTorch ≥ 2.1.0
- OpenAI API (optional, if you want to discover semantic concepts using LMMs)

1. Clone this repository adn move to the project working directory:
2. 
```shell
git clone https://github.com/OatmealLiu/FineR.git
cd FineR
```
2. Install the working environment step-by-step:
```shell
conda create -n finer python=3.9.16 -y  # create finer conda environment
conda activate finer  # activate the environment and install dependencies

pip install --upgrade pip  # enable PEP 660 support
pip install -e .  # install dependencies
pip install git+https://github.com/openai/CLIP.git  # install CLIP
```
3. If you want to discover the fine-grained semantic concepts by yourself via LLMs, please state your OpenAI key as:
```shell
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```
Note: you can add the above line to your `~/.bashrc` or `~/.zshrc` file as well

## 📦 Datasets Preparation
FineR is training-free. So we only need to prepare discovery and evaluation datasets, for which we provide the splits
used in our paper in Google Drive.
1. **Download**: we can download the 5 fine-grained datasets and the Pokemon dataset by clicking:
<p align="center">
<a href="https://drive.google.com/file/d/1iKctvfTNjYD1AEEqTIRGEqESXTVAC2Zg/view?usp=drive_link">
<strong>‍🦤 Bird-200</strong></a>
· <a href="https://drive.google.com/file/d/1pKssGj5qej5HEHXsWRsvw-oOpWiedS76/view?usp=drive_link">
<strong>🚙 Car-196</strong></a>
· <a href="https://drive.google.com/file/d/1HzVOIlNu5Tat0_fCWBGw8trx1KbqWKlH/view?usp=drive_link">
<strong>🐕 Dog-120</strong></a>
· <a href="https://drive.google.com/file/d/1iRvXYM8WhkM7M1-GLpTsJ8YWZUGDaAIE/view?usp=drive_link">
<strong>🌼 Flower-102</strong></a>
· <a href="https://drive.google.com/file/d/1kBuSXnzvh32IaYX_CowOmbzr1kQYST1T/view?usp=drive_link">
<strong>🐈 Pet-37</strong></a>
· <a href="https://drive.google.com/file/d/1F_UzN5TE-RfNnLKxykUl_EuOHKiey5L-/view?usp=sharing">
<strong>👾 Pokemon-10</strong></a>
</p>

or via `gdown`:
```shell
# Go to your datasets storage directory
cd YOUR_DATASETS_DOWNLOAD_FOLDER

# Bird-200
gdown 'https://drive.google.com/uc?id=1iKctvfTNjYD1AEEqTIRGEqESXTVAC2Zg'
# Car-196
gdown 'https://drive.google.com/uc?id=1pKssGj5qej5HEHXsWRsvw-oOpWiedS76'
# Dog-120
gdown 'https://drive.google.com/uc?id=1HzVOIlNu5Tat0_fCWBGw8trx1KbqWKlH'
# Flower-102
gdown 'https://drive.google.com/uc?id=1iRvXYM8WhkM7M1-GLpTsJ8YWZUGDaAIE'
# Pet-37
gdown 'https://drive.google.com/uc?id=1kBuSXnzvh32IaYX_CowOmbzr1kQYST1T'
# Pokemon-10
gdown 'https://drive.google.com/uc?id=1F_UzN5TE-RfNnLKxykUl_EuOHKiey5L-'
```

2. **Organize**:  we can now extract the downloaded datasets wherever we want and can organize the downloaded datasets 
via softlinks (`ln -s`):
```shell
# Go to FineR working directory and replace YOUR_DATASETS_DOWNLOAD_FOLDER with your datasets storage directory
cd FineR/datasets
sh link_local_sets.sh YOUR_DATASETS_DOWNLOAD_FOLDER
```
after which, the directory will look like and we are ready to go:
```
FineR
    └── datasets
          ├── car_196
          ├── CUB_200_2011
          ├── dogs_120
          ├── flowers_102
          ├── pet_37
          └── pokemon
```

## 📋 Evaluation
We provide all the intermediate results, including super-category, useful attributes, attribute-description pairs,
LLM-prompts, raw LLM replies, and parsed LLM replies from our disovery-->grouping pipeline under `experiments`.
So that we can directly do evaluation here.

For the experiments using **3 images per class for discovery** (paper Tab. 1), we can run them all via `sh batch_launcher_eval` 
or one-by-one:
```shell
# Bird-200
sh scripts_eval/b_pipe.sh
# Car-196
sh scripts_eval/c_pipe.sh
# Dog-120
sh scripts_eval/d_pipe.sh
# Flower-102
sh scripts_eval/f_pipe.sh
# Pet-37
sh scripts_eval/p_pipe.sh
# Pokemon
sh scripts_eval/poke_pipe.sh
```

For the experiments using **random (long-tailed) images per class for discovery** (paper Tab. 2),, we can run them all via 
`sh batch_launcher_eval_random` 
or one-by-one:
```shell
# Bird-200
sh scripts_eval_random/b_pipe.sh
# Car-196
sh scripts_eval_random/c_pipe.sh
# Dog-120
sh scripts_eval_random/d_pipe.sh
# Flower-102
sh scripts_eval_random/f_pipe.sh
# Pet-37
sh scripts_eval_random/p_pipe.sh
```

## Citation
```bibtex
@inproceedings{
    liu2024democratizing,
    title={Democratizing Fine-grained Visual Recognition with Large Language Models},
    author={Mingxuan Liu and Subhankar Roy and Wenjing Li and Zhun Zhong and Nicu Sebe and Elisa Ricci},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=c7DND1iIgb}
}
```