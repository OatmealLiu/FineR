from PIL import Image
import torchvision.transforms as transforms
import os
import random
import shutil
from copy import deepcopy
import numpy as np
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import POKEMON_STATS
import pathlib
import json

SUPERCLASS = 'pokemon'
CLASSUNIT = 'names'

pokemon_how_to1 = f"""
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

pokemon_how_to2 = f"""
Please tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS} according to the 
example of about what are the useful attributes for distinguishing bird species in a photo of a bird. Output a Python 
list object that contains the listed useful attributes.

===
Question: What are the useful attributes for distinguishing bird species in a photo of a bird?
===
Answer: ['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern',
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color',
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color',
'nape color', 'belly color', 'wing shape', 'size', 'shape',
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color',
'bill color', 'crown color', 'wing pattern', 'habitat']
===
Question: What are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}?
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


class PokemonPrompter:
    def __init__(self):
        self.supercategory = "pokemon"
        self.first_question = "general"
        self.attributes = ['color','size','shape','body pattern','number of limbs','eye shape','eye color','mouth shape','tail shape','number of heads','number of eyes','associated element','special markings','ear shape','skin texture','evolutionary stage','number of wings','leg shape','arm shape','facial expression','unique feature','body posture''tail presence','fin presence','horn presence','crest or spike presence']

    # def _generate_does_have_prompt(self, attr):
    #     return f"Questions: Does it have {attr}? Answer:"

    # def _generate_whatis_prompt(self, attr):
    #     return f"Questions: What is its {attr}? Answer:"
    #
    # def _generate_statement_prompt(self, attr):
    #     return f"Describe its {attr}."

    def _generate_does_have_prompt(self, attr):
        return f"Questions: Does the pokemon in this image have {attr}? Answer:"

    def _generate_whatis_prompt(self, attr):
        return f"Questions: What is the {attr} of the {self.supercategory} in this image? Answer:"

    def _generate_statement_prompt(self, attr):
        return f"Describe the {attr} of the {self.supercategory} in this image."

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        # list_prompts = ["Describe this image."]
        list_prompts = ["Describe the pokemon in this image."]

        for attr in self.attributes:
            if "presence" in attr:
                attr = attr.split("presence")[0].strip()
                list_prompts.append(self._generate_does_have_prompt(attr))
            else:
                list_prompts.append(self._generate_statement_prompt(attr))
        return list_prompts


    def get_llm_prompt(self, list_attr_val):
        prompt = f"""
        I have a image of a {self.supercategory} . 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} in this image from the general description and 
        attribute descriptions delimited by triple backticks with five sentences.
        2 - Infer and list three possible names of the {self.supercategory} in this photo based on the 
        information you get.
        3 - Output a JSON object that uses the following format
        <three possible pokemon names>: [
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
        Three possible pokemon names: <three possible pokemon names>
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
        - '''{list_attr_val[21][0]}''': '''{list_attr_val[21][1]}'''
        - '''{list_attr_val[22][0]}''': '''{list_attr_val[22][1]}'''
        - '''{list_attr_val[23][0]}''': '''{list_attr_val[23][1]}'''
        - '''{list_attr_val[24][0]}''': '''{list_attr_val[24][1]}'''
        - '''{list_attr_val[25][0]}''': '''{list_attr_val[25][1]}'''
        """
        return prompt


class PokemonDiscovery:
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all')

        self.class_folders = os.listdir(img_root)
        for i in range(len(self.class_folders)):
            self.class_folders[i] = os.path.join(img_root, self.class_folders[i])

        self.samples = []
        self.targets = []
        for folder in self.class_folders:
            label = int(folder.split('/')[-1][:3])
            # label = label - 1
            file_names = os.listdir(folder)

            for name in file_names:
                self.samples.append(os.path.join(folder, name))
                self.targets.append(label)

        self.classes = POKEMON_STATS['class_names']
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


def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


class PokemonDataset(Dataset):
    """
        Pokemon Dataset
    """
    def __init__(self, root, transform=None):
        self.loader = default_loader
        self.data_dir = root
        self.data = []
        self.target = []

        self.transform = transform

        meta_path = os.path.join(self.data_dir, "annotation.json")
        loaded_metas = load_json(meta_path)
        for img_path, label in loaded_metas.items():
            self.data.append(
                os.path.join(root, img_path)
            )
            self.target.append(int(label))

        self.classes = POKEMON_STATS['class_names']

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target, idx

    def __len__(self):
        return len(self.data)


def build_pokemon_prompter(cfg: dict):
    prompter = PokemonPrompter()
    return prompter


def build_pokemon_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = PokemonDiscovery(cfg['data_dir'])
    return set_to_discover


def build_pokemon_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = PokemonDataset(data_path, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


if __name__ == "__main__":
    root = "/home/miu/GoldsGym/global_datasets/pokemon"
    tfms = _transform(224)
    # cfg = {
    #     "data_dir": root,
    #     "image_size": 224,
    #     "batch_size": 36,
    #     "num_workers": 8,
    # }

    dataset = PokemonDataset(root, transform=tfms)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f'Num All Classes: {len(set(dataset.target))}')
    print(f'Len set: {len(dataset)}')
    print(f"Image {dataset.data[-1]} has Label {dataset.target[-1]} whose class name {dataset.classes[dataset.target[-1]]}")

    for file_path in dataset.data:
        if os.path.exists(file_path):
            print("The file exists.")
        else:
            print(file_path)
            print("The file does not exist.")

