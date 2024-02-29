from data.data_stats import BIRD_STATS, DOG_STATS, FLOWER_STATS, PET_STATS, CAR_STATS, POKEMON_STATS
from data.bird200 import build_bird_prompter, build_bird200_discovery, build_bird200_test, bird_how_to1, bird_how_to2, build_bird200_swav_train
from data.dog120 import build_dog_prompter, build_dog120_discovery, build_dog120_test, dog_how_to1, dog_how_to2, build_dog120_swav_train
from data.flower102 import build_flower_prompter, build_flower102_discovery, build_flower102_test, flower_how_to1, flower_how_to2, build_flower102_swav_train
from data.pet37 import build_pet_prompter, build_pet37_discovery, build_pet37_test, pet_how_to1, pet_how_to2, build_pet37_swav_train
from data.car196 import build_car_prompter, build_car196_discovery, build_car196_test, car_how_to1, car_how_to2, build_car196_swav_train
from data.pokemon import build_pokemon_prompter, build_pokemon_discovery, build_pokemon_test, pokemon_how_to1, pokemon_how_to2

from data.bird200 import _transform as bird_transform
from data.car196 import _transform as car_transform
from data.dog120 import _transform as dog_transform
from data.flower102 import _transform as flower_transform
from data.pet37 import _transform as pet_transform
from data.pokemon import _transform as pokemon_transform

from .utils import random_augmentation

__all__ = [
    "DATA_STATS", "PROMPTERS", "DATA_DISCOVERY", "DATA_GROUPING", "DATA_TRANSFORM",
    "BIRD_STATS", "DOG_STATS", "FLOWER_STATS", "PET_STATS", "CAR_STATS", "POKEMON_STATS",
    "build_bird_prompter", "build_bird200_discovery", "build_bird200_test",
    "build_dog_prompter", "build_dog120_discovery", "build_dog120_test",
    "build_flower_prompter", "build_flower102_discovery", "build_flower102_test",
    "build_pet_prompter", "build_pet37_discovery", "build_pet37_test",
    "build_car_prompter", "build_car196_discovery", "build_car196_test",
    "build_pokemon_prompter", "build_pokemon_discovery", "build_pokemon_test"
]


HOW_TOS1 = {
    "bird": bird_how_to1,
    "dog": dog_how_to1,
    "flower": flower_how_to1,
    "pet": pet_how_to1,
    "car": car_how_to1,
    "pokemon": pokemon_how_to1,
}

HOW_TOS2 = {
    "bird": bird_how_to2,
    "dog": dog_how_to2,
    "flower": flower_how_to2,
    "pet": pet_how_to2,
    "car": car_how_to2,
    "pokemon": pokemon_how_to2,
}

DATA_STATS = {
    "bird": BIRD_STATS,
    "dog": DOG_STATS,
    "flower": FLOWER_STATS,
    "pet": PET_STATS,
    "car": CAR_STATS,
    "pokemon": POKEMON_STATS,
}


PROMPTERS = {
    "bird": build_bird_prompter,
    "dog": build_dog_prompter,
    "flower": build_flower_prompter,
    "pet": build_pet_prompter,
    "car": build_car_prompter,
    "pokemon": build_pokemon_prompter,
}


DATA_DISCOVERY = {
    "bird": build_bird200_discovery,            # done
    "dog": build_dog120_discovery,
    "flower": build_flower102_discovery,
    "pet": build_pet37_discovery,
    "car": build_car196_discovery,
    "pokemon": build_pokemon_discovery,
}


DATA_GROUPING = {
    "bird": build_bird200_test,
    "dog": build_dog120_test,
    "flower": build_flower102_test,
    "pet": build_pet37_test,
    "car": build_car196_test,
    "pokemon": build_pokemon_test,
}


DATA_TRANSFORM = {
    "bird": bird_transform,
    "dog": dog_transform,
    "flower": flower_transform,
    "pet": pet_transform,
    "car": car_transform,
    "pokemon": pokemon_transform,
}


DATA_SWAV = {
    "bird": build_bird200_swav_train,
    "dog": build_dog120_swav_train,
    "flower": build_flower102_swav_train,
    "pet": build_pet37_swav_train,
    "car": build_car196_swav_train,
}

DATA_AUGMENTATION = random_augmentation


