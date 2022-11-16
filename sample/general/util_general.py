import math
import os

import pandas as pd
from haversine import haversine
from matplotlib import cm as cm, pyplot as plt

from configurations import OUTPUT_PATH

plasma = cm.get_cmap("plasma")


def get_day(t: pd.Timestamp) -> str:
    return f"{t.year}_{t.month}_{t.day}"


def calculate_distance(a, b):
    return haversine((a.lat, a.lon), (b.lat, b.lon), unit='m')


def save_fig_analysis(title, directory, file_name):
    base_path = os.path.join(OUTPUT_PATH, "data_analysis", directory)
    os.makedirs(base_path, exist_ok=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, file_name), dpi=600)
    plt.close()


def average(my_list):
    if len(my_list) == 0:
        return 0
    return sum(my_list) / len(my_list)


def std(my_list):
    avg = average(my_list)
    return math.sqrt(sum((v - avg) ** 2 for v in my_list) / len(my_list))
