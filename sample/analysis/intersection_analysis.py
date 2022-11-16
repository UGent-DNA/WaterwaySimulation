import math
from typing import Dict, List

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.graphics.correlation as sgc

from configurations import loc_eng
from sample.data_processing.data_classes import VesselTrip
from sample.general.convoy_detection import find_convoy, trips_in_convoy
from sample.general.util_general import save_fig_analysis
from sample.general.util_ships import get_ship_type, get_source_destination

matplotlib.rcParams.update({"font.size": 14})
plasma = cm.get_cmap("plasma")

sns.set(font_scale=2)
sns.set_style("whitegrid")


def get_size2(length, width):
    if length < 40:
        return "Tiny"
    if length < 101 and width < 15:
        return "Small"
    if length > 134:
        return "Long"
    return "Medium"


def get_size(length, width):
    if math.isnan(length):
        return 0
    return int(length / 20) * 20


keys = ["Intersection_time_(s)", "Route", "Route_mirror", "Ship_type", "Weekday", "Hour", "Length", "Width",
        "Depart_time",
        "Size_(m)"]
continuous_keys_for_pairplot = ["Intersection_time_(s)", "Weekday", "Hour", "Length", "Width"]


def get_dataframe(vessel_trip_list: List[VesselTrip]) -> pd.DataFrame:
    """ Transform vessel_trips to a panda dataframe for Seaborn visualisation. """
    variables: Dict[str, List] = {k: [] for k in keys}
    for vt in vessel_trip_list:
        intersection_time = sum(p[1] for p in vt.get_path() if p[0] == "Intersection")
        if intersection_time > 600:
            continue
        variables["Intersection_time_(s)"].append(intersection_time)
        s_d = get_source_destination(vt)
        s_d = (loc_eng[s_d[0]], loc_eng[s_d[1]])
        variables["Route_mirror"].append(f"{s_d[0]} --> {s_d[1]}")
        variables["Route"].append(f"{sorted(s_d)[0]} <--> {sorted(s_d)[1]}")
        variables["Ship_type"].append(get_ship_type(vt.vessel))
        variables["Weekday"].append(vt.trip[0].time.weekday())
        variables["Hour"].append(vt.trip[0].time.hour)
        variables["Length"].append(vt.vessel.length_max)
        variables["Width"].append(vt.vessel.width_max)
        variables["Depart_time"].append(vt.trip[0].time.hour * 60 + vt.trip[0].time.minute)
        variables["Size_(m)"].append(get_size(vt.vessel.length_max, vt.vessel.width_max))
    return pd.DataFrame(variables)


def get_correlations(vts: List[VesselTrip]):
    df = get_dataframe(vts)
    sgc.plot_corr(df[continuous_keys_for_pairplot].corr(), xnames=continuous_keys_for_pairplot)
    save_fig_analysis("Correlations", "regression", "correlation_plot2.png")

    sns.pairplot(df[continuous_keys_for_pairplot])
    save_fig_analysis("", "regression", "pairplot.png")

    df_type = df[["Intersection_time_(s)", "Ship_type"]]
    sns.catplot(x="Ship_type", y="Intersection_time_(s)", data=df_type)
    save_fig_analysis("Correlations", "regression", "ship_type.png")


def sideways_jitter_seaborn_image(df, to_predict, cat_1, cat_2, df_att: str):
    """ Generates a figure based on three variables

    :param df: The dataframe, should contain columns to_predict, cat_1 and cat_2
    :param to_predict: Predicted continuous variable: on x-axis
    :param cat_1: Categorical variable with up to 30 values: y-axis
    :param cat_2: Categorical variable with up to 7 values: in color code
    :param df_att: String for title to indicate changes in the dataframe (e.g., a different data selection)
    :return:
    """
    cat_2_vals = len(set(df[cat_2]))
    sort_order = sorted(set(df[cat_1]), key=lambda w: sorted(w.split("> "))) if "Route" in cat_1 \
        else sorted(set(df[cat_1]))

    g = sns.catplot(y=cat_1, x=to_predict, orient="h", height=10, aspect=2, hue=cat_2,
                    order=sort_order,
                    palette=[plasma.colors[int(i * 220 / (cat_2_vals - 1))] for i in range(cat_2_vals)],
                    data=df[[to_predict, cat_1, cat_2]])
    g.set(xlim=(0, None))
    save_fig_analysis(f"{df_att}", "regression", f"{to_predict}_{cat_1}_{cat_2}_{df_att}.png")


def get_distribution_intersection_time(df, df_att: str):
    """ Make a figure of the distribution of intersection time; test which general distribution is the best match."""
    intersection_times = list(df["Intersection_time_(s)"])
    count_dict = {}
    prec = 10
    for val in intersection_times:
        val_a = int(val / prec)
        if val_a not in count_dict:
            count_dict[val_a] = 0
        count_dict[val_a] += 1
    x_vals = sorted(count_dict.keys())

    heights = [count_dict[x] for x in x_vals]
    plt.bar(x_vals, height=heights)
    plt.xticks(ticks=range(0, 61, 10), labels=[10 * i for i in range(0, 61, 10)])
    plt.xlabel("Time in intersection in seconds")
    plt.ylabel("Number of ships")
    save_fig_analysis(f"distribution of intersection time: {df_att}", "patterns",
                      f"distribution_intersection_{df_att}.png")

    # heights[0] = 0
    # f = Fitter([i for i in intersection_times if i > 5],
    #            distributions=["gamma",
    #                           "lognorm",
    #                           "beta",
    #                           "burr",
    #                           "norm", "cauchy"])
    # f.fit()
    # print(f.summary())
    # print(f.get_best(method="sumsquare_error"))
    # save_fig(f"Distribution_test_{df_att}", "patterns", f"distribution_intersection_distr_{df_att}.png")


def make_plots(vts: List[VesselTrip]):
    get_correlations(vts)

    convoys = find_convoy(vts)
    vts_convoys = [t for t in trips_in_convoy(convoys) if t.get_path()[0][0] == "Albertkanaal"]
    for vesseltrips, name in zip([vts, vts_convoys], ["all", "convoy"]):
        dataframe = get_dataframe(vesseltrips)
        # sideways_jitter_seaborn_image(dataframe, "Depart_time", "Route_mirror", "Ship_type", name)
        # sideways_jitter_seaborn_image(dataframe, "Depart_time", "Size_(m)", "Ship_type", name)
        sideways_jitter_seaborn_image(dataframe, "Intersection_time_(s)", "Route_mirror", "Ship_type", name)
        sideways_jitter_seaborn_image(dataframe, "Intersection_time_(s)", "Size_(m)", "Ship_type", name)
        sideways_jitter_seaborn_image(dataframe, "Intersection_time_(s)", "Hour", "Ship_type", name)
        sideways_jitter_seaborn_image(dataframe, "Intersection_time_(s)", "Hour", "Weekday", name)
        sideways_jitter_seaborn_image(dataframe, "Depart_time", "Weekday", "Size_(m)", name)
        get_distribution_intersection_time(dataframe, name)
