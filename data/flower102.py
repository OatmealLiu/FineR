import os
import os.path
import torch
from PIL import Image
import torchvision.transforms as transforms
import pathlib
from typing import Any, Callable, Optional, Tuple
import numpy as np
from scipy.io import loadmat
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import FLOWER_STATS
import random
import shutil
from copy import deepcopy
from data.utils import get_swav_transform

SUPERCLASS = 'flower'
CLASSUNIT = 'categories'


flower_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}.

Specifically, you can complete the task by following the instructions below:
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in 
a photo of a bird. You should understand and learn this example carefully.
2 - List the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}.
3 - Output a Python list object that contains the listed useful attributes.

===
<bird species>
The useful attributes for distinguishing bird species in a photo of a bird:
['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern',
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color',
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color',
'nape color', 'belly color', 'wing shape', 'size', 'shape',
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color',
'bill color', 'crown color', 'wing pattern', 'habitat']
===

===
<{SUPERCLASS} {CLASSUNIT}>
The useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}:
===
"""

flower_how_to2 = f"""
Please tell me what are the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS} \
according to the example of about what are the useful visual attributes for distinguishing bird species in a photo of a bird. \
Output a Python list object that contains the listed useful visual attributes.

===
Question: What are the useful visual attributes for distinguishing bird species in a photo of a bird?
===
Answer: ['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern',
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color',
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color',
'nape color', 'belly color', 'wing shape', 'size', 'shape',
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color',
'bill color', 'crown color', 'wing pattern', 'habitat']
===
Question: What are the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}?
===
Answer:
"""


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def make_dataset(dir, image_ids, targets):
    assert (len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'jpg',
                             '%s' % image_ids[i]), targets[i])
        images.append(item)
    return images


class FlowerPrompter:
    def __init__(self):
        self.supercategory = "flower"
        self.first_question = "general"
        self.howmany_attr = ['number of petals']

        self.attributes = ['primary flower color','flower size','flower color gradient','flower center pattern','flower color intensity','flower color variation','flower shape','flower pattern','flower arrangement','flower center color','flower center shape',  'flower symmetry','petal color','petal pattern','petal color pattern','petal color intensity','petal color variation','petal color gradient','petal shape','petal size','number of petals','petal symmetry','petal arrangement','petal texture','stem length','stem color','stem texture','stem pattern','stem thickness','leaf shape','leaf color','leaf arrangement','leaf texture','leaf margin','leaf venation','leaf size']

    def _generate_howmany_prompt(self, attr):
        return f"Questions: How many {attr} does the {self.supercategory} in this photo have? Answer:"

    def _generate_whatis_prompt(self, attr):
        return f"Questions: What is the {attr} of the {self.supercategory} in this photo? Answer:"

    def _generate_statement_prompt(self, attr):
        return f"Describe the {attr} of the {self.supercategory} in this photo."

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        list_prompts = ["Describe this image in details."]

        for attr in self.attributes:
            if attr in self.howmany_attr:
                list_prompts.append(self._generate_howmany_prompt(attr))
            else:
                list_prompts.append(self._generate_statement_prompt(attr))

        return list_prompts

    def get_llm_prompt(self, list_attr_val):
        prompt = f"""
        I have a photo of a {self.supercategory} commonly occuring in the United Kingdom. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} from the general description and \
        attribute descriptions delimited by triple backticks with five sentences.
        2 - Infer and list three possible {self.supercategory} category names of the \
        {self.supercategory} in this photo based on the information you get.
        3 - Output a JSON object that uses the following format
        <three possible {self.supercategory} category names>: [
                <first sentence of the summary>,
                <second sentence of the summary>,
                <third sentence of the summary>,
                <fourth sentence of the summary>,
                <fifth sentence of the summary>,
        ]

        Use the following format to perform the aforementioned tasks:
        General Description: '''general description of the photo'''
        Attributes List:
        - '''attribute name''': '''attribute description'''
        - '''attribute name''': '''attribute description'''
        - ...
        - '''attribute name''': '''attribute description'''
        Summary: <summary>
        Three possible {self.supercategory} category names: <three possible {self.supercategory} category names>
        Output JSON: <output JSON object>

        '''{list_attr_val[0][0]}''': '''{list_attr_val[0][1]}'''
        Attributes List:
        - '''{list_attr_val[1][0]}''': '''{list_attr_val[1][1]}'''
        - '''{list_attr_val[2][0]}''': '''{list_attr_val[2][1]}'''
        - '''{list_attr_val[3][0]}''': '''{list_attr_val[3][1]}'''
        - '''{list_attr_val[4][0]}''': '''{list_attr_val[4][1]}'''
        - '''{list_attr_val[5][0]}''': '''{list_attr_val[5][1]}'''
        - '''{list_attr_val[6][0]}''': '''{list_attr_val[6][1]}'''
        - '''{list_attr_val[7][0]}''': '''{list_attr_val[7][1]}'''
        - '''{list_attr_val[8][0]}''': '''{list_attr_val[8][1]}'''
        - '''{list_attr_val[9][0]}''': '''{list_attr_val[9][1]}'''
        - '''{list_attr_val[10][0]}''': '''{list_attr_val[10][1]}'''
        - '''{list_attr_val[11][0]}''': '''{list_attr_val[11][1]}'''
        - '''{list_attr_val[12][0]}''': '''{list_attr_val[12][1]}'''
        - '''{list_attr_val[13][0]}''': '''{list_attr_val[13][1]}'''
        - '''{list_attr_val[14][0]}''': '''{list_attr_val[14][1]}'''
        - '''{list_attr_val[15][0]}''': '''{list_attr_val[15][1]}'''
        - '''{list_attr_val[16][0]}''': '''{list_attr_val[16][1]}'''
        - '''{list_attr_val[17][0]}''': '''{list_attr_val[17][1]}'''
        - '''{list_attr_val[18][0]}''': '''{list_attr_val[18][1]}'''
        - '''{list_attr_val[19][0]}''': '''{list_attr_val[19][1]}'''
        - '''{list_attr_val[20][0]}''': '''{list_attr_val[20][1]}'''
        - '''number of {list_attr_val[21][0]}''': '''{list_attr_val[21][1]}'''
        - '''{list_attr_val[22][0]}''': '''{list_attr_val[22][1]}'''
        - '''{list_attr_val[23][0]}''': '''{list_attr_val[23][1]}'''
        - '''{list_attr_val[24][0]}''': '''{list_attr_val[24][1]}'''
        - '''{list_attr_val[25][0]}''': '''{list_attr_val[25][1]}'''
        - '''{list_attr_val[26][0]}''': '''{list_attr_val[26][1]}'''
        - '''{list_attr_val[27][0]}''': '''{list_attr_val[27][1]}'''
        - '''{list_attr_val[28][0]}''': '''{list_attr_val[28][1]}'''
        - '''{list_attr_val[29][0]}''': '''{list_attr_val[29][1]}'''
        - '''{list_attr_val[30][0]}''': '''{list_attr_val[30][1]}'''
        - '''{list_attr_val[31][0]}''': '''{list_attr_val[31][1]}'''
        - '''{list_attr_val[32][0]}''': '''{list_attr_val[32][1]}'''
        - '''{list_attr_val[33][0]}''': '''{list_attr_val[33][1]}'''
        - '''{list_attr_val[34][0]}''': '''{list_attr_val[34][1]}'''
        - '''{list_attr_val[35][0]}''': '''{list_attr_val[35][1]}'''
        - '''{list_attr_val[36][0]}''': '''{list_attr_val[36][1]}'''
        """
        return prompt


class FlowerDiscovery102:
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all{folder_suffix}')

        self.class_folders = os.listdir(img_root)  # 100 x 1
        for i in range(len(self.class_folders)):
            self.class_folders[i] = os.path.join(img_root, self.class_folders[i])

        self.samples = []
        self.targets = []
        for folder in self.class_folders:
            label = int(folder.split('/')[-1][:3])
            # label = label - 1
            file_names = os.listdir(folder)

            for name in file_names:
                self.targets.append(label)
                self.samples.append(os.path.join(folder, name))

        self.classes = FLOWER_STATS['class_names']
        self.index = 0

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.samples):
            raise StopIteration
        img = Image.open(self.samples[self.index]).convert("RGB")
        target = self.targets[self.index]
        self.index += 1
        return img, target


class FlowerDataset(Dataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
            self,
            root: str,
            train=False,
            transform: Optional[Callable]=None,
            loader=default_loader
    ) -> None:
        self.splits = ('train', 'val', 'trainval', 'test')

        if train:
            split = "trainval"
        else:
            split = "test"

        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))

        self.split = split
        self.root = root

        self.transform = transform
        self.loader = loader

        self._base_folder = pathlib.Path(self.root)
        self._images_folder = self._base_folder / "jpg"

        if self.split == 'trainval':
            set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
            image_ids_train = set_ids[self._splits_map['train']].tolist()
            image_ids_val = set_ids[self._splits_map['val']].tolist()
            image_ids = image_ids_train + image_ids_val

        else:
            set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
            image_ids = set_ids[self._splits_map[self.split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

        image_name = [f"image_{image_id:05d}.jpg" for image_id in image_ids]

        samples = make_dataset(self.root, image_name, self._labels)  # 2040
        self.samples = samples
        self.uq_idxs = np.array(range(len(self)))

        self.data = [path for path, label in self.samples]
        self.target = [label for path, label in self.samples]
        self.classes = FLOWER_STATS['class_names']

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[idx]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, target, self.uq_idxs[idx]
        return sample, target, path     # just for visualization

    def extra_repr(self) -> str:
        return f"split={self.split}"


def build_flower_prompter(cfg: dict):
    prompter = FlowerPrompter()
    return prompter


def build_flower102_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = FlowerDiscovery102(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_flower102_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = FlowerDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_flower102_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = FlowerDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def create_discovery_set(root, files, targets, class_names, num_per_category=3, selection='largest'):
    # selection = 'random' or 'largest'
    target_image_dict = {i: [] for i in range(len(class_names))}
    cleaned_names = deepcopy(class_names)
    cleaned_names = [name.replace(' ', '_') for name in cleaned_names]
    cleaned_names = [name.replace('-', '_') for name in cleaned_names]

    folder_template = "{}.{}"
    file_template = "{}.{}_{}"

    for label, image in zip(targets, files):
        target_image_dict[label].append(image)

    num_copied = 0
    for label, images in target_image_dict.items():
        image_paths = deepcopy(images)

        if selection == 'random':
            random.shuffle(image_paths)
        else:
            image_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
            image_paths = image_paths[:50]
            random.shuffle(image_paths)

        selected_images = image_paths[:num_per_category]

        destination_folder = folder_template.format(str(label).zfill(3), cleaned_names[label])
        destination_folder = os.path.join(root, destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for src_img in selected_images:
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('/')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


def generate_long_tail_distribution(num_categories=102):
    import numpy as np
    import matplotlib.pyplot as plt

    categories = num_categories

    # Generate power-law distributed random numbers.
    # The parameter 'a' controls the shape of the distribution.
    # Smaller 'a' gives a heavier tail.
    a = 2.0
    samples = np.random.zipf(a, categories)

    # Adjust samples to ensure they're within the range 1 to 10
    adjusted_samples = np.clip(samples, 1, 10)

    # Sort samples in descending order
    sorted_samples = np.sort(adjusted_samples)[::-1]

    print(sorted_samples)

    # Optional: Plot to see the distribution
    plt.bar(range(categories), sorted_samples)
    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images across Categories')
    plt.show()


def create_long_tail_discovery_set(root, files, targets, class_names, selection='largest'):
    distribution = [
        10, 10, 10, 10, 10, 10,  9,  8,  8,  6,  6,  5,  5,  4,  4,  3,  3,
        3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
    ]

    # selection = 'random' or 'largest'
    target_image_dict = {i: [] for i in range(len(class_names))}
    cleaned_names = deepcopy(class_names)
    cleaned_names = [name.replace(' ', '_') for name in cleaned_names]
    cleaned_names = [name.replace('-', '_') for name in cleaned_names]

    folder_template = "{}.{}"
    file_template = "{}.{}_{}"

    for label, image in zip(targets, files):
        target_image_dict[label].append(image)

    num_copied = 0
    for label, images in target_image_dict.items():
        num_per_category = distribution[label]

        image_paths = deepcopy(images)

        if selection == 'random':
            random.shuffle(image_paths)
        else:
            image_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
            image_paths = image_paths[:50]
            random.shuffle(image_paths)

        selected_images = image_paths[:num_per_category]

        destination_folder = folder_template.format(str(label).zfill(3), cleaned_names[label])
        destination_folder = os.path.join(root, destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for src_img in selected_images:
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('/')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


if __name__ == "__main__":
    root = "/home/miu/GoldsGym/global_datasets/flowers_102"
    tfms = _transform(224)

    dataset = FlowerDataset(root, train=True, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f'Num All Classes: {len(set(dataset.target))}')
    print(f"Num All Images: {len(dataset.data)}")

    print(f'Len set: {len(dataset)}')
    print(f"Image {dataset.data[-1]} has Label {dataset.target[-1]} whose class name {dataset.classes[dataset.target[-1]]}")

    # print("Create random unlabeled set as wild discovery set")
    # num_per_category_list = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    # # num_per_category_list = [3]
    #
    # for num_per_category in num_per_category_list:
    #     root_disco = os.path.join(root, f"images_discovery_all_{num_per_category}")
    #
    #     if not os.path.exists(root_disco):
    #         os.makedirs(root_disco)
    #
    #     create_discovery_set(root_disco, dataset.data, dataset.target, dataset.classes,
    #                          num_per_category=num_per_category, selection='largest')

    # print("Create long-tailed unlabeled set as wild discovery set")
    # root_disco = os.path.join(root, f"images_discovery_all_random")
    #
    # if not os.path.exists(root_disco):
    #     os.makedirs(root_disco)
    #
    # create_long_tail_discovery_set(
    #     root_disco, dataset.data, dataset.target, dataset.classes, selection='largest'
    # )