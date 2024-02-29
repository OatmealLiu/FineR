# Can you help me to optimize the following code to let it plot the 5 figures in 1 figure with only 1 row. I want to insert this figure into a A4  sized paper.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Adjust font sizes using rcParams
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


num_car = 196
car_longtail = [
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

num_dog = 120
dog_longtail = [
    10, 10, 10, 10, 10, 10, 10, 9, 8, 7, 7, 6, 6, 5, 5, 5, 5,
    4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1
]

num_flower = 102
flower_longtail = [
        10, 10, 10, 10, 10, 10,  9,  8,  8,  6,  6,  5,  5,  4,  4,  3,  3,
        3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
    ]

num_pet = 37
pet_longtail = [
    10, 10, 10, 10, 10, 10, 8, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2,
    2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1
]

num_bird = 200
bird_longtail = [
    10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 7, 7, 7, 5, 5, 5,
    5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]


def plot_long_tail_distribution(num_categories, dist_longtail, save_pdf=None):
    categories = range(num_categories)

    # Generating gradient color: from deep indigo of space to the light yellow of stars
    color_start = np.array([75/255, 0/255, 130/255])  # Indigo
    color_end = np.array([255/255, 223/255, 186/255])  # Light yellow (akin to a distant star's glow)
    colors = [color_start + (color_end - color_start) * i / num_categories for i in categories]

    # Plotting
    plt.bar(categories, dist_longtail, color=colors)
    plt.xlabel('Categories')
    plt.ylabel('Number of Images for Reasoning')
    plt.title('Distribution of Images across Categories')

    # Save plot as PDF if required
    if save_pdf:
        plt.savefig(save_pdf, format='pdf')
        plt.close()
    else:
        plt.show()


def plot_combined_long_tail_distributions(datasets, save_pdf=None):
    fig, axs = plt.subplots(2, 3, figsize=(12.27, 8.27), tight_layout=True)  # 2 rows, 3 columns, with square shape

    color_start = np.array([75/255, 0/255, 130/255])  # Indigo
    color_end = np.array([255/255, 223/255, 186/255])  # Light yellow

    # Find the maximum number of categories across all datasets
    max_categories = max([num for num, _ in datasets])

    # Flatten axs for easy iteration
    axs = axs.ravel()

    for i, (num_categories, dist_longtail) in enumerate(datasets):
        colors = [color_start + (color_end - color_start) * j / num_categories for j in range(num_categories)]
        axs[i].bar(range(num_categories), dist_longtail, color=colors)
        axs[i].set_xlim(0, num_categories)  # Set consistent x-axis limit
        axs[i].set_xticks([num_categories])
    # Remove the last, unused subplot
    fig.delaxes(axs[-1])

    if save_pdf:
        plt.savefig(save_pdf, format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # plot_long_tail_distribution(num_bird, bird_longtail, 'plots/bird_disco-set_longtail.pdf')
    # plot_long_tail_distribution(num_car, car_longtail, 'plots/car_disco-set_longtail.pdf')
    # plot_long_tail_distribution(num_dog, dog_longtail, 'plots/dog_disco-set_longtail.pdf')
    # plot_long_tail_distribution(num_flower, flower_longtail, 'plots/flower_disco-set_longtail.pdf')
    # plot_long_tail_distribution(num_pet, pet_longtail, 'plots/pet_disco-set_longtail.pdf')

    datasets = [
        (num_bird, bird_longtail),
        (num_car, car_longtail),
        (num_dog, dog_longtail),
        (num_flower, flower_longtail),
        (num_pet, pet_longtail)
    ]
    plot_combined_long_tail_distributions(datasets, 'plots/combined_disco-set_longtail.pdf')
