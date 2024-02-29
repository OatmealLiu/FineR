from PIL import Image
import torchvision.transforms as transforms
import os
import random
import shutil
from copy import deepcopy
import numpy as np
from scipy import io as mat_io
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import CAR_STATS
import pathlib
from data.utils import get_swav_transform


SUPERCLASS = 'car'
CLASSUNIT = 'models'


car_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing specific {SUPERCLASS} {CLASSUNIT} \
(for example: 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008','Acura TSX Sedan 2012', \
'Acura Integra Type R 2001','Acura ZDX Hatchback 2012','Aston Martin V8 Vantage Convertible 2012', \
'Aston Martin V8 Vantage Coupe 2012','Aston Martin Virage Convertible 2012','Aston Martin Virage Coupe 2012', \
'Audi RS 4 Convertible 2008','Audi A5 Coupe 2012','Audi TTS Coupe 2012','Audi R8 Coupe 2012', \
'Audi V8 Sedan 1994','Audi 100 Sedan 1994','Audi 100 Wagon 1994','Audi TT Hatchback 2011','Audi S6 Sedan 2011', \
'Audi S5 Convertible 2012','Audi S5 Coupe 2012','Audi S4 Sedan 2012','Audi S4 Sedan 2007',Audi TT RS Coupe 2012') \ 
in a photo of a {SUPERCLASS}. \

Specifically, you can complete the task by following the instructions below: \
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in \
a photo of a bird. You should understand and learn this example carefully. \
2 - List the useful attributes for distinguishing specific {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}. \
3 - Output a Python list object that contains the listed useful attributes. \

=== \
<bird species> \
The useful attributes for distinguishing bird species in a photo of a bird: \
['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern', \
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color', \
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color', \
'nape color', 'belly color', 'wing shape', 'size', 'shape', \
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color', \
'bill color', 'crown color', 'wing pattern', 'habitat'] \
=== \

=== \
<{SUPERCLASS} {CLASSUNIT}> \
The useful attributes for distinguishing specific {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}: \
=== \
"""

car_how_to2 = f"""
Please tell me what are the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} from its appearance \
in a photo, like the example of about what are the useful visual attributes for distinguishing bird species in a \
photo I give you later. List the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo and \
output a Python list object that contains the listed useful visual attributes. \

=== \
Question: What are the useful visual attributes for distinguishing bird species in a photo of a bird? \
=== \
Answer: ['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern', \
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color', \
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color', \
'nape color', 'belly color', 'wing shape', 'size', 'shape', \
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color', \
'bill color', 'crown color', 'wing pattern', 'habitat'] \
=== \
Question: What are the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}? \
=== \
Answer: \
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


class CarPrompter:
    def __init__(self):
        self.supercategory = "car"
        self.superunit = 'model'

        self.first_question = "general"
        self.attributes = ['approximate year of manufacture (1990s, 2000s or 2010s)','possible make (automobile manufacturers)','doors','seats','windows','body style (Sedan, Wagon, SUV, Coupe, Roadster, Truck, Cab, Convertible, Minivan, Van, Hatchback, etc.)','body color','roof color','size','height','length','width','window size','window shape','window tint''emblem/logo on the front or the rear of the car','emblem/logo placement''grille design','grille shape','grille size','distinctive elements of the grille','headlight design','headlight shape','headlight size','taillight design','taillight shape','taillight size','wheel design','wheel size','wheel pattern','specific body panels, contours, or accent lines''roofline shape','door handle design','side mirror design','bumper design','hood design']

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
        # a series of questions
        list_prompts = ["Describe this image in details."]

        for attr in self.attributes:
            if attr in ['doors', 'seats', 'windows']:
                list_prompts.append(self._generate_howmany_prompt(attr))
            elif attr in ['approximate year of manufacture (1990s, 2000s or 2010s)', 'model', 'make (automobile manufacturers)']:
                list_prompts.append(self._generate_whatis_prompt(attr))
            else:
                list_prompts.append(self._generate_statement_prompt(attr))

        return list_prompts

    def get_llm_prompt(self, attr_descr_pairs):
        prompt = f"""
        I have a photo of a {self.supercategory}. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} from the general description and \
        attribute descriptions delimited by triple backticks with five sentences.
        2 - The description might not be correct and accurate. So I need you to infer and list three possible detailed car model names (e.g., Mercedes-Benz C-Class Sedan 2012) of the \
        {self.supercategory} in this photo based on the information you get.
        3 - Output a JSON object that uses the following format
        <three possible detailed car model names>: [
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
        Three possible detailed car model names: <three possible detailed car model names>
        Output JSON: <output JSON object>

        '''{attr_descr_pairs[0][0]}''': '''{attr_descr_pairs[0][1]}'''
        Attributes List:
        - '''{attr_descr_pairs[1][0]}''': '''{attr_descr_pairs[1][1]}'''
        - '''{attr_descr_pairs[2][0]}''': '''{attr_descr_pairs[2][1]}'''
        - '''number of {attr_descr_pairs[3][0]}''': '''{attr_descr_pairs[3][1]}'''
        - '''number of {attr_descr_pairs[4][0]}''': '''{attr_descr_pairs[4][1]}'''
        - '''number of {attr_descr_pairs[5][0]}''': '''{attr_descr_pairs[5][1]}'''
        - '''{attr_descr_pairs[6][0]}''': '''{attr_descr_pairs[6][1]}'''
        - '''{attr_descr_pairs[7][0]}''': '''{attr_descr_pairs[7][1]}'''
        - '''{attr_descr_pairs[8][0]}''': '''{attr_descr_pairs[8][1]}'''
        - '''{attr_descr_pairs[9][0]}''': '''{attr_descr_pairs[9][1]}'''
        - '''{attr_descr_pairs[10][0]}''': '''{attr_descr_pairs[10][1]}'''
        - '''{attr_descr_pairs[11][0]}''': '''{attr_descr_pairs[11][1]}'''
        - '''{attr_descr_pairs[12][0]}''': '''{attr_descr_pairs[12][1]}'''
        - '''{attr_descr_pairs[13][0]}''': '''{attr_descr_pairs[13][1]}'''
        - '''{attr_descr_pairs[14][0]}''': '''{attr_descr_pairs[14][1]}'''
        - '''{attr_descr_pairs[15][0]}''': '''{attr_descr_pairs[15][1]}'''
        - '''{attr_descr_pairs[16][0]}''': '''{attr_descr_pairs[16][1]}'''
        - '''{attr_descr_pairs[17][0]}''': '''{attr_descr_pairs[17][1]}'''
        - '''{attr_descr_pairs[18][0]}''': '''{attr_descr_pairs[18][1]}'''
        - '''{attr_descr_pairs[19][0]}''': '''{attr_descr_pairs[19][1]}'''
        - '''{attr_descr_pairs[20][0]}''': '''{attr_descr_pairs[20][1]}'''
        - '''{attr_descr_pairs[21][0]}''': '''{attr_descr_pairs[21][1]}'''
        - '''{attr_descr_pairs[22][0]}''': '''{attr_descr_pairs[22][1]}'''
        - '''{attr_descr_pairs[23][0]}''': '''{attr_descr_pairs[23][1]}'''
        - '''{attr_descr_pairs[24][0]}''': '''{attr_descr_pairs[24][1]}'''
        - '''{attr_descr_pairs[25][0]}''': '''{attr_descr_pairs[25][1]}'''
        - '''{attr_descr_pairs[26][0]}''': '''{attr_descr_pairs[26][1]}'''
        - '''{attr_descr_pairs[27][0]}''': '''{attr_descr_pairs[27][1]}'''
        - '''{attr_descr_pairs[28][0]}''': '''{attr_descr_pairs[28][1]}'''
        - '''{attr_descr_pairs[29][0]}''': '''{attr_descr_pairs[29][1]}'''
        - '''{attr_descr_pairs[30][0]}''': '''{attr_descr_pairs[30][1]}'''
        - '''{attr_descr_pairs[31][0]}''': '''{attr_descr_pairs[31][1]}'''
        - '''{attr_descr_pairs[32][0]}''': '''{attr_descr_pairs[32][1]}'''
        - '''{attr_descr_pairs[33][0]}''': '''{attr_descr_pairs[33][1]}'''
        """
        return prompt


class CarDiscovery196:
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

        self.classes = CAR_STATS['class_names']
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


class CarDataset(Dataset):# done
    """
        Cars Dataset
    """
    def __init__(self, root, train=True, transform=None, limit=0):
        if train:
            data_dir = "cars_train/"
            meta_path = "devkit/cars_train_annos.mat"
        else:
            data_dir = "cars_test/"
            meta_path = "devkit/cars_test_annos_withlabels.mat"

        data_dir = os.path.join(root, data_dir)
        metas = os.path.join(root, meta_path)

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0] - 1)   # Miu: Note, Scar original annotation is from 1 ~ 196
                                                    # therefore, - 1

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

        self.classes = CAR_STATS['class_names']

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        # return image, target, idx
        return image, target, self.data[idx]  # just for visualization

    def __len__(self):
        return len(self.data)


def build_car_prompter(cfg: dict):
    prompter = CarPrompter()
    return prompter


def build_car196_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = CarDiscovery196(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_car196_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = CarDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_car196_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = CarDataset(data_path, train=False, transform=tfms)

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


def generate_long_tail_distribution(num_categories=196):
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
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,  9,  8,  6,
        6,  6,  6,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  3,
        3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1
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
    root = "/home/miu/GoldsGym/global_datasets/car_196"
    tfms = _transform(224)
    # cfg = {
    #     "data_dir": root,
    #     "image_size": 224,
    #     "batch_size": 36,
    #     "num_workers": 8,
    # }

    dataset = CarDataset(root, train=True, transform=tfms)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f'Num All Classes: {len(set(dataset.target))}')
    print(f"Num All Images: {len(dataset.data)}")

    print(f'Len set: {len(dataset)}')
    print(f"Image {dataset.data[-1]} has Label {dataset.target[-1]} whose class name {dataset.classes[dataset.target[-1]]}")

    # print("Create random unlabeled set as wild discovery set")
    # num_per_category_list = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    # # num_per_category_list = [3]
    # for num_per_category in num_per_category_list:
    #     root_disco = os.path.join(root, f"images_discovery_all_{num_per_category}")
    #
    #     if not os.path.exists(root_disco):
    #         os.makedirs(root_disco)
    #
    #     create_discovery_set(root_disco, dataset.data, dataset.target, dataset.classes,
    #                          num_per_category=num_per_category,
    #                          selection='largest')

    # print("Create long-tailed unlabeled set as wild discovery set")
    # root_disco = os.path.join(root, f"images_discovery_all_random")
    #
    # if not os.path.exists(root_disco):
    #     os.makedirs(root_disco)
    #
    # create_long_tail_discovery_set(
    #     root_disco, dataset.data, dataset.target, dataset.classes, selection='largest'
    # )



