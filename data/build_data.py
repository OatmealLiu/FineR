from __future__ import print_function
from torch.utils.data import DataLoader, Subset
import pathlib


def build_prompt_generator(cfg: dict):
    if cfg['dataset_name'] == 'cub':
        from .bird200 import CUBAttributes as PromptGenerator
    else:
        from .bird200 import CUBAttributes as PromptGenerator

    attr_prompt_generator = PromptGenerator()
    return attr_prompt_generator


def build_train_base(cfg: dict):
    if cfg['dataset_name'] == 'cub':
        from .bird200 import CUBBase100 as BaseData
    else:
        from .bird200 import CUBBase100 as BaseData

    dataloader = BaseData(cfg['data_dir'])
    return dataloader


def build_train_novel(cfg: dict):
    if cfg['dataset_name'] == 'cub':
        from .bird200 import CUBNovel100 as NovelData
    else:
        from .bird200 import CUBNovel100 as NovelData

    dataloader = NovelData(cfg['data_dir'])
    return dataloader


def build_data(cfg: dict):
    if cfg['dataset_name'] == 'cub':
        from .bird200 import CUBDataset as Dataset
        from .bird200 import _transform as transform
    elif cfg['dataset_name'] == 'dog':
        from .dog120 import DogDataset as Dataset
        from .dog120 import _transform as transform
    elif cfg['dataset_name'] == 'car':
        from .car196 import CarDataset as Dataset
        from .car196 import _transform as transform
    elif cfg['dataset_name'] == 'pet':
        from .pet37 import PetDataset as Dataset
        from .pet37 import _transform as transform
    elif cfg['dataset_name'] == 'flower':
        from .flower102 import FlowerDataset as Dataset
        from .flower102 import _transform as transform
    else:
        raise NameError(f"{cfg['dataset_name']} dataset does not exist!")


    data_path = pathlib.Path(cfg['data_dir'])
    tfms = transform(cfg['image_size'])

    dataset = Dataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


