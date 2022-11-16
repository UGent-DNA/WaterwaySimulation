import math
import os
import pathlib
import random
import warnings
from typing import Dict, List, Tuple, Iterator

import dill
import matplotlib
import seaborn as sns
import pandas as pd
from matplotlib import cm as cm, pyplot as plt

import configurations as conf
from configurations import OUTPUT_PATH, table_extension
from sample.data_processing.data_classes import VesselTrip
from sample.general.interval_manipulation import time_to_queue_length, total_interval_time, Interval
from sample.general.table_writer import TableWriter
from sample.general.util_general import average, std
from sample.simulation.tripstate import TripState

plasma = cm.get_cmap("plasma").colors
matplotlib.rcParams.update({'font.size': 14})



def show_save(name):
    # plt.tight_layout()
    path = os.path.join(OUTPUT_PATH, name)
    plt.savefig(path + '.png', format='png', dpi=600, bbox_inches="tight")
    plt.close()


def read_dict(filename):
    res = {}
    for line in open(filename):
        k, v = line.strip().split(" ")
        res[str(k)] = int(v)
    return res


def plot_dict(total_interval_times, title):
    x = sorted(total_interval_times.keys())
    plt.figure()
    ax = plt.gca()
    ax.bar(x, [total_interval_times[k] for k in x])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax.set_xticklabels(x, rotation=90)
    plt.title(title)
    show_save(os.path.join("calibration", title))


# Calibration
def run_calibration():
    type_of_day = "midweek"
    for root, dirs, files in os.walk(os.path.join(OUTPUT_PATH, "calibration", "logs")):
        originals = []
        experiments = {}
        for filename in sorted(files):
            if type_of_day not in filename:
                continue

            filepath = os.path.join(root, filename)
            total_interval_times = read_dict(filepath)
            if "original" in filename:
                originals.append(total_interval_times)
            else:
                experiments[filename] = total_interval_times

            # plot_dict(total_interval_time, filename)

        original_summary = {}
        for d in originals:
            for k, v in d.items():
                original_summary.setdefault(k, []).append(v)
        original_summary = {k: sum(v) / len(v) for k, v in original_summary.items()}
        plot_dict(original_summary, f"original_summary_{type_of_day}")

        for name, experiment in experiments.items():
            diff_plot = {}
            diffs = []
            diffs_squared = []
            for k, v in experiment.items():
                diff = original_summary[k] - v
                diffs.append(diff)
                diffs_squared.append(diff ** 2)
                diff_plot[k] = diff
            print(f"{name:25s} {sum(abs(d) for d in diffs):7.0f} {sum(diffs_squared):15.0f}")
            plot_dict(diff_plot, f"{name}_diff")


def read_queue_intervals(filepath) -> Iterator[Tuple[Dict[int, Dict[str, List[Tuple[int, int]]]],
                                                     Dict[int, List[VesselTrip]],
                                                     Dict[int, Dict[Interval, str]]]]:
    setup_to_location = {}
    for root, dirs, files in os.walk(os.path.join(filepath, "logs")):
        root_path = pathlib.PurePath(root)
        setup = root_path.name
        setup_to_location[setup] = (root, files)
    sorted_setups = sorted(setup_to_location.keys())
    print(sorted_setups)

    for setup in sorted_setups:
        root, files = setup_to_location[setup]
        # if "200" not in setup.split('_') or "weekend" in setup.split('_'):
        #     continue
        if files:
            queue_intervals = {}
            vessel_trips = {}
            schedule = {}
        else:
            continue
        for filename in sorted(files):
            name = filename.split(".")[0]
            if len(name) > 4 and "vessel" in name:
                iteration = int(name.split('_')[-1])
                if iteration > conf.n_iterations:
                    continue
                vessel_trips[iteration] = dill.load(open(os.path.join(root, filename), "rb"))
                continue
            if len(name) > 4 and "signal" in name:
                iteration = int(name.split('_')[-1])
                if iteration > conf.n_iterations:
                    continue
                schedule[iteration] = dill.load(open(os.path.join(root, filename), "rb"))
                continue

            iteration = int(name)
            if iteration > conf.n_iterations:
                continue

            with open(os.path.join(root, filename)) as f:
                loc_to_queue_intervals = {}
                for line in f:
                    loc, k, v = line.strip().split(" ")
                    loc_to_queue_intervals.setdefault(loc, []).append((int(k), int(v)))

            queue_intervals[iteration] = loc_to_queue_intervals

        yield setup, queue_intervals, vessel_trips, schedule


def get_time(time_in_seconds):
    h = math.floor(time_in_seconds / 3600)
    m = int(math.floor(time_in_seconds - h * 3600) / 60)
    return h, m


def run_queue_over_time():
    """ Generate a table with average wait times per scenario and plots. Based on queue_over_time logs. """
    filepath_base = os.path.join(OUTPUT_PATH, conf.save_location)
    # queue_intervals, _, schedules = read_queue_intervals(filepath_base)
    wait_times_all = []
    table_rows = []

    queue_timings_all = {}

    for setup, queue_intervals, vessel_trips, schedules in read_queue_intervals(filepath_base):
        print(setup)
        if "base" in setup:
            continue
        endpoints = {"AK1_canal", "AK5_harbor"} if "AK5" in setup else {"AK1_canal", "AK3_harbor"}

        loc_set = set(queue_intervals[0].keys())
        colors = {loc: plasma[int(i * 255 / len(loc_set))] for i, loc in
                  enumerate(sorted(loc_set, key=lambda loc: loc[::-1]))}

        queue_timings = {}
        wait_times = []
        for iteration, loc_to_queue_intervals in queue_intervals.items():
            loc_to_queue_times = {loc: time_to_queue_length(time_to_length) for loc, time_to_length in
                                  loc_to_queue_intervals.items()}

            schedule = schedules[iteration]
            total_wait_time = sum(total_interval_time(qi) for qi in loc_to_queue_intervals.values())
            wait_times.append(total_wait_time)

            queue_timings[iteration] = _get_queue_timings(endpoints, loc_to_queue_times, schedule)

            visuals = True
            if visuals and iteration == 0 and conf.generate_figures_time_to_queue:
                plot_time_to_queue_length(colors, endpoints, filepath_base, loc_to_queue_times, schedule, setup,
                                          total_wait_time)

        wait_times_all += wait_times
        queue_timings_all[setup] = queue_timings
        setup_list = setup.split('_')[1:]
        table_rows.append([setup_list[0], f"{setup_list[1]:3s}", f"{setup_list[2]:3s}", f"{setup_list[3]:2s}",
                           f"{'-'.join(setup_list[4:]):23s}",
                           f"{average(wait_times):7.0f}",
                           f"{average(wait_times) / 3600:7.0f}", f"({std(wait_times) / 3600:.0f})"])
        colors = {"AK1_canal": [0.050383, 0.029803, 0.527975], "AK5_harbor": [0.981826, 0.618572, 0.231287],
                  "AK3_harbor": [0.981826, 0.618572, 0.231287]}

        if conf.generate_figures_queue_unresolved:
            plot_queues_not_resolved(colors, endpoints, filepath_base, queue_intervals, queue_timings, setup)
        table_queues_start_sizes(endpoints, queue_intervals, queue_timings, setup)

    header = ["day", "\\#trips", "OW", "sea", "model", "wait (s)", "wait (h)", "std (h)"]
    TableWriter("wait_times").write_table(header, table_rows, table_extension)

    print(f"|{average(wait_times_all):7.0f}s | {average(wait_times_all) / 3600:7.0f}h | {std(wait_times_all):7.0f}|")
    return queue_timings_all


def time_string(h, m):
    r = ""
    if h < 10:
        r += "0"
    r += f"{h}:"
    if m < 10:
        r += "0"
    r += str(m)
    return r


def plot_queues_not_resolved(colors, endpoints, filepath_base, queue_intervals, queue_timings, setup):
    plt.figure()
    ax = plt.gca()
    all_x = []
    for j, loc in enumerate(endpoints):
        count_dict = {}
        for iteration, loc_to_queue_intervals in queue_intervals.items():
            t_list = [qt[0] for qt in queue_timings[iteration][loc] if qt[2] != 0]
            for t in t_list:
                count_dict.setdefault(t, 0)
                count_dict[t] += 1
        x = sorted(count_dict.keys())
        all_x += x
        y = [count_dict[x_] for x_ in x]
        width = 1500
        ax.bar([x_ for x_ in x], y, width=width, label=loc, color=colors[loc])
    # if all_x:
    # tick_labels = [i for i in range(math.ceil(min(all_x) / 3600), math.floor(max(all_x) / 3600) + 1)]
    ax.set_xticks([3600 * i for i in range(3, 24, 3)], range(3, 24, 3))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Queues not empty after direction switch (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 24 * 3600)
    ax.legend()
    filename = os.path.join(filepath_base, f"queues_unresolved_{setup}")
    show_save(filename)


def table_queues_start_sizes(endpoints, queue_intervals, queue_timings, setup):
    for j, loc in enumerate(endpoints):
        queue_start_size_by_time: Dict[int, List[int]] = {}
        count_dict = {}
        count_dict2 = {}

        for iteration, loc_to_queue_intervals in queue_intervals.items():
            for qt in queue_timings[iteration][loc]:
                t = qt[0]
                queue_start_size_by_time.setdefault(t, []).append(qt[1])
                if qt[2] != 0:
                    count_dict.setdefault(t, 0)
                    count_dict[t] += 1

                if qt[3] > 60 * 15:
                    count_dict2.setdefault(t, 0)
                    count_dict2[t] += 1

        x = sorted(queue_start_size_by_time.keys())

        rows = [[f"{time_string(*get_time(x_))}",
                 # f"{len(queue_start_size_by_time[x_]):3d}",
                 f"{average(queue_start_size_by_time[x_]):5.2f}",
                 f"{std(queue_start_size_by_time[x_]):5.2f}",
                 f"{max(queue_start_size_by_time[x_]):3d}",
                 f"{int(100 * count_dict.get(x_, 0) / len(queue_start_size_by_time[x_])):3d}",
                 f"{int(100 * count_dict2.get(x_, 0) / len(queue_start_size_by_time[x_])):3d}"]
                for x_ in x]
        tw = TableWriter(f"queue_start_size_table_{setup}_{loc}")
        header = ["time", "avg", "std", "max", "\\#Q NE (\\%)", "Q>15m"]
        tw.write_table(header, rows, table_extension)


def _get_queue_timings(endpoints, loc_to_queue_times, schedule) -> Dict[str, List[Tuple[int, int, int, int]]]:
    qt = {}
    for loc, time_to_length in loc_to_queue_times.items():
        qt[loc] = []
        if loc not in endpoints:
            continue
        mins, maxs = _schedule_mins_maxs(loc, schedule, time_to_length, True)
        for mi, ma in zip(mins, maxs):
            qt[loc].append((ma[0], ma[1], mi[1], mi[0] - ma[0]))
    return qt


def plot_time_to_queue_length(colors, endpoints, filepath_base, loc_to_queue_times, schedule, setup, total_wait_time):
    min_ = min([min(t_l, default=0) for t_l in loc_to_queue_times.values()], default=0)
    max_ = max([max(t_l, default=0) for t_l in loc_to_queue_times.values()], default=0)
    if max_ == 0:
        return
    tick_labels = [i for i in range(math.ceil(min_ / 3600), math.floor(max_ / 3600) + 1)]
    x = list(range(min_, max_ + 1))
    fig = plt.figure(figsize=(20, 7))
    ax = plt.gca()
    if schedule:
        _add_schedule_bar(ax, colors, schedule)
    for loc, time_to_length in loc_to_queue_times.items():
        ax.plot(x, [time_to_length.get(i, 0) for i in x],
                linestyle='dashed' if 'canal' in loc else 'solid',
                label=loc, color=colors[loc])

        if loc not in endpoints:
            continue
        mins, maxs = _schedule_mins_maxs(loc, schedule, time_to_length, start_queue=False)
        # for mi, ma in zip(mins, maxs):
        #     h, m = get_time(ma[0])
        #     print(f"{h :2d}:{m:2d} {ma[1]:2d} --> {mi[1]:2d} in {(mi[0] - ma[0]) / 60:6.2f}m")

        ax.scatter([q[0] for q in maxs], [q[1] for q in maxs], color='b')
        ax.scatter([q[0] for q in mins], [q[1] for q in mins], color='b', marker='s')
    ax.set_xticks([3600 * i for i in tick_labels], tick_labels)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Ships queueing to enter location")
    ax.set_ylim(-.8, 25)
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"time_to_queue_length_{setup}"
    plt.title(f"Total queue time: {total_wait_time / 60:4.0f}m")
    plt.savefig(os.path.join(filepath_base, filename))
    plt.close(fig)


def _schedule_mins_maxs(loc, schedule, time_to_length, start_queue):
    mins = []
    maxs = []
    for iv, signal in schedule.items():
        for direction in ["canal", "harbor"]:
            if direction in loc and signal == direction and int(iv[1]) != int(iv[0]):
                r = list(range(int(iv[0]), int(iv[1])))
                index_min = min(r, key=lambda i: time_to_length.get(i, 0))
                index_max = max(r, key=lambda i: time_to_length.get(i, 0))
                mins.append((index_min, time_to_length.get(index_min, 0)))
                if start_queue:
                    maxs.append((int(iv[0] / (15 * 60)) * 15 * 60, time_to_length.get(int(iv[0]), 0)))
                else:
                    maxs.append((index_max, time_to_length.get(index_max, 0)))
    return mins, maxs


def _add_schedule_bar(ax, cols, schedule):
    color_map = {"blocked": plasma[120], "canal": cols["AK1_canal"],
                 "harbor": cols["AK5_harbor"] if "AK5_harbor" in cols else cols.get("AK3_harbor", 180),
                 "free": plasma[240]}
    patch_handles = []
    for (start, end), signal in schedule.items():
        d = end - start
        patch_handles.append(ax.barh(0, d, height=1.6, color=color_map[signal], align='center', left=start))
    for j, signal in enumerate(schedule.values()):
        for i, patch in enumerate(patch_handles[j].get_children()):
            bl = patch.get_xy()
            x_ = 0.5 * patch.get_width() + bl[0]
            y_ = 0.5 * patch.get_height() + bl[1]
            ax.text(x_, y_, signal[0].upper(), ha='center')


def get_results(setup, queue_intervals, vessel_trips, schedules):
    area = {"AK1", "AK2", "AK3"}
    if "AK5" in setup.split("_"):
        area.update({"Fork_narrow", "AK4", "AK5"})
    endpoints = {"AK1_canal", "AK5_harbor"} if "AK5" in setup else {"AK1_canal", "AK3_harbor"}
    # wait_times = []

    number_trips = []
    total_sailing_time = []
    max_total_queue_time = []
    max_total_queue_length = []
    for (iteration, trip_list), (_, loc_to_queue_intervals) in zip(vessel_trips.items(), queue_intervals.items()):
        max_total_queue_time.append(max(sum(p[2] for p in vt.get_path()) for vt in trip_list))

        tst = []
        nst = []
        for vt in trip_list:
            path = vt.get_path()
            tst.append(sum(p[1] + get_queue_time(i, path) for i, p in enumerate(path)))
            nst.append(sum(p[1] + get_queue_time(i, path) for i, p in enumerate(path) if p[0] in area))
            for i, p in enumerate(path):
                vt.path[i] = p[:2]
        t, _, vt = max((t, random.random(), vt) for t, vt in zip(nst, trip_list))

        loc_to_queue_times = {"_".join(loc): time_to_queue_length(time_to_length) for loc, time_to_length in
                              loc_to_queue_intervals.items()}

        schedule = schedules[iteration]
        # total_wait_time = sum(total_interval_time(qi) for qi in loc_to_queue_intervals.values())
        # wait_times.append(total_wait_time)

        qt = _get_queue_timings(endpoints, loc_to_queue_times, schedule)

        list_ql = qt["AK3_harbor"] if len(area) == 3 else qt["AK5_harbor"]
        max_total_queue_length.append(
            (max((ql[1] for ql in qt["AK1_canal"]), default=0), max((ql[1] for ql in list_ql), default=0)))

        total_sailing_time.append(sum(tst))

        number_trips.append(len(trip_list))

    ship_travel_times = [st / n for st, n in zip(total_sailing_time, number_trips)]
    avg_ship_travel_time = average(ship_travel_times)
    canal_max_q = average([m[0] for m in max_total_queue_length])
    harbor_max_q = average([m[1] for m in max_total_queue_length])
    ql = max(canal_max_q, harbor_max_q)

    return avg_ship_travel_time, ql


def run_tests_time(queues_by_time):
    filepath_base = os.path.join(OUTPUT_PATH, conf.save_location)
    # _, vessel_trips, _ = read_queue_intervals(filepath_base)
    table_rows = []
    visualise_queue_by_direction = True
    pareto_plot_dict = {}
    for setup, queue_intervals, vessel_trips, schedule in read_queue_intervals(filepath_base):
        trip_directions = {}
        bulk_minutes = 60

        number_trips = []
        total_sailing_time = []
        narrow_sailing_time = []
        max_narrow_sailing = []
        max_total_queue_time = []
        max_total_queue_length = []
        for iteration, trip_list in vessel_trips.items():
            area = {"AK1", "AK2", "AK3"}
            if "AK5" in setup.split("_"):
                area.update({"Fork_narrow", "AK4", "AK5"})

            max_total_queue_time.append(max(sum(p[2] for p in vt.get_path()) for vt in trip_list))

            tst = []
            nst = []
            for vt in trip_list:
                path = vt.get_path()
                tst.append(sum(p[1] + get_queue_time(i, path) for i, p in enumerate(path)))
                nst.append(sum(p[1] + get_queue_time(i, path) for i, p in enumerate(path) if p[0] in area))
                for i, p in enumerate(path):
                    vt.path[i] = p[:2]
            t, _, vt = max((t, random.random(), vt) for t, vt in zip(nst, trip_list))

            if "base" not in setup:
                qt = queues_by_time[setup][iteration]
                list_ql = qt["AK3_harbor"] if len(area) == 3 else qt["AK5_harbor"]
                max_total_queue_length.append((max(ql[1] for ql in qt["AK1_canal"]), max(ql[1] for ql in list_ql)))

            total_sailing_time.append(sum(tst))
            narrow_sailing_time.append(sum(nst))
            max_narrow_sailing.append((t, vt))

            number_trips.append(len(trip_list))

            for trip in trip_list:
                t = trip.trip[0].time.time()
                trip_start = int((t.minute + t.hour * 60) / bulk_minutes)
                direction = TripState(trip).get_direction()

                trip_directions.setdefault(direction,
                                           [[0] * int(24 * 60 / bulk_minutes) for _ in range(len(vessel_trips))])
                trip_directions[direction][iteration][trip_start] += 1

        if visualise_queue_by_direction:
            plt.figure(figsize=(20, 7))
            ax = plt.gca()
            x = list(range(0, 24))
            transposed_canal = [[n_trips[x_] for n_trips in trip_directions["canal"]] for x_ in x]
            transposed_harbor = [[n_trips[x_] for n_trips in trip_directions["harbor"]] for x_ in x]
            plt.errorbar([x_ - 0.075 for x_ in x], [average(transposed_canal[x_]) for x_ in x],
                         yerr=[std(transposed_canal[x_]) for x_ in x], capsize=4,
                         label='Canal', color=plasma[50])
            plt.errorbar([x_ + 0.075 for x_ in x], [average(transposed_harbor[x_]) for x_ in x],
                         yerr=[std(transposed_harbor[x_]) for x_ in x], capsize=4,
                         label='Harbor', color=plasma[200])

            # plt.plot([x_ - 0.2 for x_ in x], [average(transposed_canal[x_]) for x_ in x], color=plasma[50])
            # plt.plot([x_ + 0.2 for x_ in x], [average(transposed_harbor[x_]) for x_ in x], color=plasma[200])

            ax.set_xticks([int(i * len(x) / 24) for i in range(24)], x)
            ax.set_xlabel("Time of day (h)")
            ax.set_ylabel("Number of trips")
            ax.legend()
            # plt.tight_layout(rect=[0, 0, 1, 0.95])
            filename = f"queue_by_direction_{setup}"
            # plt.title(f"Total queue time: {total_wait_time / 60:4.0f}m")
            show_save(filename)
            visualise_queue_by_direction = False

        avg_ship_travel_time = [st / n for st, n in zip(total_sailing_time, number_trips)]
        avg_ship_narrow_sailing = [st / n for st, n in zip(narrow_sailing_time, number_trips)]
        mns_t, mns_vt = max(max_narrow_sailing, key=lambda m: m[0])
        # print(f"|{setup:50s} | {average(number_trips):4.0f}-({std(number_trips):2.0f}) "
        #       f"|{average(total_sailing_time):7.0f}-({std(total_sailing_time):5.0f}) "
        #       f"|{average(avg_ship_travel_time):6.0f}-({std(avg_ship_travel_time):3.0f})"
        #       f"|{average(avg_ship_narrow_sailing):6.0f}-({std(avg_ship_narrow_sailing):3.0f})"
        #       f"| {mns_t}|")
        canal_max_q = average([m[0] for m in max_total_queue_length])
        harbor_max_q = average([m[1] for m in max_total_queue_length])
        ql = max(canal_max_q, harbor_max_q)
        pareto_plot_dict[setup] = (average(avg_ship_travel_time), ql)

        setup_list = setup.split('_')[1:]
        table_rows.append([setup_list[0],
                           f"{setup_list[1]:3s}",
                           # f"{setup_list[2]:3s}",
                           f"{setup_list[3]:2s}",
                           f"{'-'.join(setup_list[4:]):23s}",
                           f"{average(avg_ship_travel_time):5.0f}",  # -({std(avg_ship_travel_time):5.0f})",
                           f"{std(avg_ship_travel_time):5.0f}",
                           # f"{average(avg_ship_narrow_sailing):5.0f}",  # -({std(avg_ship_narrow_sailing):5.0f})",
                           # f"{mns_t:4d}",
                           # f"{average(max_total_queue_time) / 60:4.0f}",
                           f"{canal_max_q:.0f}",
                           f"{harbor_max_q:.0f}",
                           f"{average(avg_ship_travel_time) + 150 * ql:6.0f}"])
    header = ["day", "\\#trips", "sea", "model", "ST", "std ST", "Q canal", "Q harbor", "score"]
    # header = ["day", "\\#trips", "OW", "sea", "model", "ST", "std ST", "narrow ST", "max narrow ST", "max QT (m)",
    #           "Q canal", "Q harbor"]
    TableWriter("simulator_summarized_results").write_table(header, table_rows, table_extension)

    model_name_map = {"Reg15x15": "R15_15", "Dynamic": "Dyn", "Reg30x15": "R30_15",
                      "Reg45x15": "R45_15", "RegOpt": "R_opt"}

    mkeys = list(model_name_map.keys())
    markers = ['o', 's', 'd', 'v', 'p', 'X']

    for ntrips in ["150", "200", "250", "300"]:
        for single_direction in ["AK3", "AK5"]:
            plt.figure(figsize=(7, 7))
            ax = plt.gca()
            # x = list(range(max(v[1] for v in pareto_plot_dict.values())))

            for s, (st, ql) in pareto_plot_dict.items():
                if ntrips not in s or single_direction not in s:
                    continue

                sea_status = int(s.split('_')[4].strip()[1])
                model = '-'.join(s.split('_')[5:])

                model_index = mkeys.index(model)
                if sea_status == 0:
                    if model_index == 0:
                        label = f"Sea_F {model_name_map[model]}"
                    else:
                        label = f"{model_name_map[model]}"
                elif model_index == 0:
                    label = f"Sea_T"
                else:
                    label = "_"
                marker = markers[model_index % len(markers)]
                plt.scatter(ql, st, s=72, color=plasma[sea_status * 60 + 30], marker=marker, label=label)

            # ax.set_xticks([int(i * len(x) / 24) for i in range(24)], x)
            ax.set_xlabel("Maximal queue length (any direction)")
            ax.set_ylabel("Average passage time (s)")
            ax.legend()
            filename = os.path.join(conf.save_location, f"pareto_{single_direction}_{ntrips}")
            show_save(filename)

    plt.figure()
    ax = plt.gca()
    cat1 = []
    cat2 = []
    to_predict = []
    used_models = set()
    x_tick_labels = []
    for s, (st, ql) in pareto_plot_dict.items():
        sea_status = int(s.split('_')[4].strip()[1])

        model = '-'.join(s.split('_')[5:])
        model_index = mkeys.index(model)
        ntrips = int(s.split('_')[2])

        marker = markers[model_index % len(markers)]

        score = int((ql + 150 * st)/1000)

        scenario = f"{ntrips}{'T' if sea_status == 2 else 'F'}"
        cat1.append(scenario)
        cat2.append(model_name_map[model])
        to_predict.append(score)
        if scenario not in x_tick_labels:
            x_tick_labels.append(scenario)

        if model == "Dynamic":
            continue

        if model not in used_models:
            plt.scatter(scenario, score, s=72, color=plasma[model_index * 40 + 30], marker=marker,
                        label=model_name_map[model])
            used_models.add(model)
        else:
            plt.scatter(scenario, score, s=72, color=plasma[model_index * 40 + 30], marker=marker)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Score (/1000)")
    plt.xticks(rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    filename = os.path.join(conf.save_location, f"all_model_comparison")
    show_save(filename)
    # path = os.path.join(OUTPUT_PATH, filename)
    # plt.savefig(path + '.png', format='png', dpi=600, bbox_inches="tight")
    # plt.close()


def get_queue_time(i, path):
    return path[i + 1][2] if i + 1 < len(path) else 0

# if __name__ == "__main__":
#     queues_by_time = run_queue_over_time()
#     run_tests_time(queues_by_time)
