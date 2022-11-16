import math
import os
from typing import List, Tuple, Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance

from configurations import days, months
from sample.data_processing.data_classes import VesselTrip, VesselState
from sample.data_processing.read_data import get_day, get_filtered_trips
from sample.general.util_general import plasma, save_fig_analysis
from sample.general.util_ships import get_ship_type, get_source_destination
from sample.simulation.tripstate import TripState

matplotlib.rcParams.update({'font.size': 14})

Interval = Tuple[float, float]


def get_ship_distributions(plot_distributions=False, ship_counter="number") -> Dict[pd.Timestamp, List[float]]:
    """ Count the number (or sum the time) of vessels that pass by day and hour.

    :param plot_distributions: if True, generate a plot for each day
    :param ship_counter: can be number or time
    :return: Match a day-timestamp to a 24-element list containing the number (of time) of ships that pass that day and
    hour
    """
    ship_distributions = {}
    for month in [3, 4]:
        for day in range(1, 32):
            if month == 4 and day > 24:
                continue
            ships_per_hour, time_eval = _plot_ship_distributions_by_location(day, month, plot_distributions,
                                                                             ship_counter,
                                                                             get_filtered_trips([(month, day)]))
            ship_distributions[time_eval] = ships_per_hour
    return ship_distributions


def _plot_ship_distributions_by_location(day, month, plot_distributions, ship_counter, vessel_trips_o) \
        -> Tuple[List[float], pd.Timestamp]:
    """ Generate a distribution for a single day, color by starting location and destination. """
    ships_per_hour_in_p = [0] * 24
    ships_per_hour_out_p = [0] * 24
    time_eval = 0
    if ship_counter == "number":
        plt.ylim(0, 25)
    else:
        plt.ylim(0, 20000)

    for i, ship_loc in enumerate(["Albertkanaal", "Albertdok", "Amerikadok", "Stadshaven", "Diverse"]):
        vessel_trips_in = [vt for vt in vessel_trips_o if get_source_destination(vt)[0] == ship_loc]
        vessel_trips_out = [vt for vt in vessel_trips_o if get_source_destination(vt)[1] == ship_loc]

        if ship_counter == "number":
            ships_per_hour_in = ships_by_xminute_count(vessel_trips_in)
            ships_per_hour_out = ships_by_xminute_count(vessel_trips_out)
        elif ship_counter == "time":
            ships_per_hour_in = ships_by_hour_time(vessel_trips_in)
            ships_per_hour_out = ships_by_hour_time(vessel_trips_out)
        else:
            raise ValueError()

        time_eval = vessel_trips_o[0].trip[0].time.floor(freq='D')

        x_vals_in = [i + 0.2 for i in range(24)]
        x_vals_out = [i - 0.2 for i in range(24)]
        if plot_distributions:
            plt.bar(x_vals_in, height=ships_per_hour_in, width=0.35, bottom=ships_per_hour_in_p, label=f"{ship_loc}",
                    color=plasma.colors[int(i * 255 / 5)])
            plt.bar(x_vals_out, height=ships_per_hour_out, width=0.35, bottom=ships_per_hour_out_p,
                    color=plasma.colors[int(i * 255 / 5)])
        for j, h in enumerate(ships_per_hour_in):
            ships_per_hour_in_p[j] += h
        for j, h in enumerate(ships_per_hour_out):
            ships_per_hour_out_p[j] += h

    plt.xticks(range(0, 24, 3), range(0, 24, 3))
    plt.xlabel("Hour of the day")
    plt.ylabel("Number of ships" if ship_counter == "number" else "Ships moving (s)")
    plt.legend()
    sd = str(day)
    filename = f"distribution_{ship_counter}_ship_type_{month}_{sd if len(sd) == 2 else '0' + sd}.png"
    base_path = os.path.join("results_by_day", f"{ship_counter}_ships")
    save_fig_analysis(f"{days[time_eval.weekday()]} {day} {months[month - 1]}", base_path, filename)
    return ships_per_hour_in_p, time_eval


def _plot_ship_distributions_by_type(day, month, plot_distributions, ship_counter, vessel_trips_o):
    """ Generate a distribution for a single day, color by ship-type. """
    ships_per_hour = [0] * 24
    time_eval = 0
    if ship_counter == "number":
        plt.ylim(0, 25)
    else:
        plt.ylim(0, 20000)
    for i, ship_type in enumerate(["Vrachtship", "Diverse", "Toerisme"]):
        vessel_trips = [vt for vt in vessel_trips_o if get_ship_type(vt.vessel) == ship_type]
        if not vessel_trips:
            continue

        if ship_counter == "number":
            ships_per_hour_c = ships_by_xminute_count(vessel_trips)
        elif ship_counter == "time":
            ships_per_hour_c = ships_by_hour_time(vessel_trips)
        else:
            raise ValueError()

        time_eval = vessel_trips[0].trip[0].time.floor(freq='D')

        if plot_distributions:
            plt.bar(list(range(24)), height=ships_per_hour_c, bottom=ships_per_hour, label=ship_type,
                    color=plasma.colors[int(i * 255 / 3)])
        for j, h in enumerate(ships_per_hour_c):
            ships_per_hour[j] += h
    plt.legend()
    sd = str(day)
    filename = f"distribution_num_ships_{month}_{sd if len(sd) == 2 else '0' + sd}.png"
    base_path = os.path.join("results_by_day", f"{ship_counter}_ship_type")
    save_fig_analysis(f"{days[time_eval.weekday()]} {day} {months[month - 1]}", base_path, filename)
    return ships_per_hour, time_eval


def round_to_multiple(x, multiple):
    return multiple * math.floor(x / multiple)


def round_to_time(vs: VesselState, minute_x: int):
    return math.floor((vs.time.hour * 60 + round_to_multiple(vs.time.minute, minute_x)) / minute_x)


def ships_by_xminute_count(vessel_trips, minute_x=60) -> List[int]:
    """ Count the number of vessels that pass each hour. """
    minute_x = round(60 / round(60 / minute_x))
    start_end = [(round_to_time(vt.trip[0], minute_x), round_to_time(vt.trip[-1], minute_x)) for vt in vessel_trips]
    ships_per_hour_c = [0] * 24 * int(60 / minute_x)
    for s, e in start_end:
        ships_per_hour_c[s] += 1
    return ships_per_hour_c


def get_ships_direction_by_time() -> Dict[str, Dict[pd.Timestamp, List[int]]]:
    trips_direction = {"neither": {}, "canal": {}, "harbor": {}, "both": {}}
    for month in [3, 4]:
        for day in range(1, 32):
            if month == 4 and day > 24:
                continue
            ftrips = get_filtered_trips([(month, day)])

            time_eval = ftrips[0].trip[0].time.floor(freq='D')

            trips_direction_i = {}
            for vt in ftrips:
                d = TripState(vt).get_direction()
                if d not in trips_direction_i:
                    trips_direction_i[d] = []
                trips_direction_i[d].append(vt)

            for d, trips in trips_direction_i.items():
                trips_direction[d][time_eval] = ships_by_xminute_count(trips, 60)

    return trips_direction


def ships_by_hour_time(vessel_trips) -> List[float]:
    """ Sum the time each vessel spends moving, separated by hour. """
    shiptime_in_seconds_per_hour_c = [0] * 24
    for vt in vessel_trips:
        hour_to_time_dict = {}
        for vs in vt.trip:
            hour = vs.time.hour
            if hour not in hour_to_time_dict:
                hour_to_time_dict[hour] = 0
            # TODO: Verify precision is 3 seconds
            hour_to_time_dict[hour] += 3
            shiptime_in_seconds_per_hour_c[hour] += 3
        # vt_hours.append((vt, hour_to_time_dict))
    return shiptime_in_seconds_per_hour_c


def weekday_analysis(ship_distributions, ship_counter, identifier="0"):
    """ Run analysis for distributions by day of the week; compare distributions and generate figures. """
    pattern_weekday_general = pattern_by_day_type(ship_distributions, range(5))
    pattern_weekend_general = pattern_by_day_type(ship_distributions, [5, 6])

    time_to_dist = {time_e: distance.jensenshannon(pattern_weekday_general, ship_hours)
                    for time_e, ship_hours in ship_distributions.items()}
    for time_e, dist in sorted(time_to_dist.items(), key=lambda d: sum(ship_distributions[d[0]])):
        print(f"{dist:5.2f} {sum(ship_distributions[time_e]):3d} {days[time_e.weekday()]:12s} {get_day(time_e):12s}")

    pattern_day = [pattern_by_day_type(ship_distributions, [i]) for i in range(7)]
    plt.figure(figsize=(11, 6), dpi=600)
    x_vals = list(range(len(pattern_weekday_general)))

    for i, p_day in enumerate(pattern_day):
        if ship_counter == "number":
            plt.ylim(0, 20)
        elif ship_counter == "time":
            plt.ylim(0, 15000)
        # Make sure the weekend pops
        color = plasma.colors[int((i + (2 if i > 4 else 0)) * 255 / 8)]
        plt.scatter(x_vals, p_day, color=color, label=days[i])
    plt.xticks(range(0, 24, 3), range(0, 24, 3))
    plt.plot(x_vals, pattern_weekday_general, color=plasma.colors[int(2 * 255 / 8)], label="Average weekday")
    plt.plot(x_vals, pattern_weekend_general, color=plasma.colors[int(7.5 * 255 / 8)], label="Average weekend")
    plt.xlabel("Hour of the day")
    plt.ylabel("Average number of ships" if ship_counter == "number" else "Average ships moving (s)")
    plt.legend()
    save_fig_analysis("Pattern by days", "patterns", f"day_of_the_week_{ship_counter}_{identifier}.png")

    time_to_dist_by_weekday = []
    for weekday in range(7):
        time_to_dist_by_weekday.append({time_e: distance.jensenshannon(pattern_day[weekday], ship_hours)
                                        for time_e, ship_hours in ship_distributions.items()
                                        if time_e.weekday() == weekday})

    for weekday in range(7):
        print()
        print(f"{days[weekday]:12s}")
        total_number_of_ships = []
        for time_e, dist in sorted(time_to_dist_by_weekday[weekday].items(), key=lambda d: d[1]):
            number_of_ships = sum(ship_distributions[time_e])
            total_number_of_ships.append(number_of_ships)
            print(f"{dist:5.2f} {number_of_ships:3d} {get_day(time_e):12s}")
        print(f"Average number of ships: {sum(total_number_of_ships) / len(total_number_of_ships):4.0f}")

    markers = ["o", "v", "^", "s", "D", "P", "."]
    weeks = sorted({t.week for t in ship_distributions.keys()})
    for week in weeks:
        used_days = []
        y = [0] * 7
        x = list(range(7))

        for t, distrib in ship_distributions.items():
            if t.week != week:
                continue
            y[t.weekday()] = sum(distrib)
            used_days.append(t)

        while y[0] == 0:
            y = y[1:]
            x = x[1:]
        while y[-1] == 0:
            y = y[:-1]
            x = x[:-1]

        color_val = plasma.colors[int(255 * (week - min(weeks)) / len(weeks))]
        plt.scatter(x, y, color=color_val, label=f"{min(used_days).day}/{min(used_days).month}",
                    marker=markers[week % len(markers)])

    plt.legend(title="Week of", bbox_to_anchor=(1, 1), loc="upper left")
    plt.xticks(range(7), [d[:3] for d in days])
    plt.ylabel("Number of ship trips")
    save_fig_analysis("Trips by day", "patterns", f"trips_by_day_{identifier}.png")


def pattern_by_day_type(ship_distributions, days_to_make_pattern):
    """ Generate pattern figures for each day of the week. """
    weekday_ship_hours = {t: s for t, s in ship_distributions.items() if t.weekday() in days_to_make_pattern}
    pattern_weekday_general = [0] * max(len(ship_h) for _, ship_h in weekday_ship_hours.items())
    for i, (t, ship_h) in enumerate(weekday_ship_hours.items()):
        if len(days_to_make_pattern) < 2:
            plt.scatter(list(range(len(ship_h))), ship_h, color=plasma.colors[int(i * 255 / len(weekday_ship_hours))],
                        label=f"{t.day}/{t.month}")
        for j, v in enumerate(ship_h):
            pattern_weekday_general[j] += ship_h[j]
    if len(days_to_make_pattern) < 2:
        plt.legend()
        plt.xticks(range(0, 24, 3), range(0, 24, 3))
        plt.ylabel("Average ships moving (s)")
        plt.xlabel("Hour of the day")
        filename = f"day_{'_'.join([days[d] for d in days_to_make_pattern])}.png"
        save_fig_analysis(f"Pattern data for {days[days_to_make_pattern[0]]}", "patterns", filename)
    for i in range(24):
        pattern_weekday_general[i] /= len(weekday_ship_hours)
    return pattern_weekday_general


def print_trip(vt: VesselTrip):
    trip_path = vt.get_path()
    start_time = f"{vt.trip[0].time.hour:2d}:{vt.trip[0].time.minute:2d}"
    print(
        f"{vt.get_duration_in_seconds():5.0f} {get_day(vt.trip[0].time):10s} {start_time:10s} "
        f"{vt.vessel.v_id:10d} "
        f"{vt.vessel.ship_type:25s} {vt.vessel.length_max:5.1f} {vt.vessel.width_max:5.1f} "
        f"{trip_path[0][0] :20s} {trip_path[-1][0]:20s} {trip_path}")


if __name__ == "__main__":
    weekday_analysis(get_ship_distributions(), "number", "all")
