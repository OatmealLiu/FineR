#!/bin/bash

# You can enter the folder where you save the data from the terminal to soft-link it to the current repo.
YOUR_DATASETS_DOWNLOAD_FOLDER=$1


ln -s YOUR_DATASETS_DOWNLOAD_FOLDER/CUB_200_2011 CUB_200_2011

ln -s YOUR_DATASETS_DOWNLOAD_FOLDER/car_196 car_196

ln -s YOUR_DATASETS_DOWNLOAD_FOLDER/dogs_120 dogs_120

ln -s YOUR_DATASETS_DOWNLOAD_FOLDER/flowers_102 flowers_102

ln -s YOUR_DATASETS_DOWNLOAD_FOLDER/pet_37 pet_37

ln -s YOUR_DATASETS_DOWNLOAD_FOLDER/pokemon pokemon

