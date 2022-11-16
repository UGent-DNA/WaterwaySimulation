import math
import os
from typing import List, Tuple, Dict

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

import configurations as conf
from configurations import RESOURCE_PATH, OUTPUT_PATH, LOCATIONS
from sample.data_processing.data_classes import VesselState, VesselTrip
from sample.general.convoy_detection import find_convoy
from sample.general.util_general import plasma

matplotlib.rcParams.update({'font.size': 14})


def scale_to_img(lat_lon, h_w):
    """ Conversion from latitude and longitude to image pixels. """
    points = (51.2529, 4.3842, 51.2271, 4.4539)
    y = ((lat_lon[0] - points[2]) * (h_w[1] - 0) / (points[0] - points[2]))
    x = ((lat_lon[1] - points[1]) * (h_w[0] - 0) / (points[3] - points[1]))
    return int(x), int(y)


def transform_points(gps_data: List[Tuple[float, float]], img_size: Tuple[float, float]):
    img_points = []
    for d in gps_data:
        x1, y1 = scale_to_img(d, img_size)
        img_points.append((x1, y1))
    return img_points


class Node:
    def __init__(self, vs: VesselState, vt: VesselTrip):
        self.vs: VesselState = vs
        self.vt: VesselTrip = vt

        self.dists: List[Tuple[float, Node]] = []


def get_time_to_vs(trips: List[VesselTrip], precision) -> Dict[pd.Timestamp, List[Node]]:
    """ List all GPS-instances based on the time of occurrence and link them to their VesselTrip.

    :param trips: List of VesselTrips.
    :param precision: Round-off in seconds. If precision is 60, there will be an entry for each minute.
    """
    time_to_vs = {}
    for node in [Node(vs, vt) for vt in trips for vs in vt.trip]:
        t_new = node.vs.time.round(f"{precision}S")
        node.vs.time = t_new
        if t_new not in time_to_vs:
            time_to_vs[t_new] = []
        if node.vt.vessel.v_id not in {n.vt.vessel.v_id for n in time_to_vs[t_new]}:
            time_to_vs[t_new].append(node)

    return time_to_vs


def get_ship_patch(base, length, width, rotation_degrees, color):
    """ Create a patch in the correct dimensions for each ship to create a pleasing visual. """
    pointiness = 1.75
    vertices = [(0, -width / 2), (length / pointiness, -width / 2), (length, 0), (length / pointiness, width / 2),
                (0, width / 2), (0, -width / 2)]
    vertices = Affine2D().rotate_deg(rotation_degrees).transform(vertices)
    vertices = [(base[0] + v[0], base[1] + v[1]) for v in vertices]
    path = Path(vertices, None)
    patch = PathPatch(path, facecolor=color, edgecolor=(0.0, 0.0, 0.0))
    return patch


class AnimateFleet:
    """ Visualise the movement of ships in the specified area. """

    def __init__(self, vessels: List[VesselTrip], precision, writename: str):
        self.x_mod = 2.518 * 1000
        self.y_mod = 1.482 * 1000
        self.fig = plt.figure(figsize=(self.x_mod / 100, self.y_mod / 100), dpi=960 / 8)
        self.ax = plt.gca()

        # all_states = [vs for vessel in vessels for vs in vessel.trip]
        self.time_to_vs = get_time_to_vs(vessels, precision)
        self.times = sorted(self.time_to_vs.keys())

        self.convoy_timings: Dict[pd.Timestamp, List[Tuple[int, int]]] = {}
        for convoy in find_convoy(vessels):
            for t, tup in convoy.get_time_to_mmsi(precision).items():
                if t not in self.convoy_timings:
                    self.convoy_timings[t] = [tup]
                else:
                    self.convoy_timings[t].append(tup)

        self.im = plt.imread(os.path.join(RESOURCE_PATH, 'map.png'))
        self.active_patches: List[PathPatch] = []

        print(f"Start image generation. Number of frames: {len(self.times)}")

        anime = FuncAnimation(
            fig=self.fig,
            func=self.update,
            init_func=self.init_ani,
            frames=len(self.times),
            interval=100
        )
        # Save as an MP4 with ffmpeg
        if self.times:
            ffmpegwriter = FFMpegWriter(fps=10)
            base_path = os.path.join(OUTPUT_PATH, "movies")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            anime.save(os.path.join(base_path, f'fleet_{writename}.mp4'), writer=ffmpegwriter)

    def init_ani(self):
        boundary = (51.2529, 4.3842, 51.2271, 4.4539)
        x_labels = [f"{boundary[1] + i * (boundary[3] - boundary[1]) / 5:5.3f}" for i in range(6)]
        self.ax.set_xticks([int(i * self.x_mod / 5) for i in range(6)], x_labels)

        y_labels = [f"{boundary[2] + i * (boundary[0] - boundary[2]) / 5:5.3f}" for i in range(6)]
        self.ax.set_yticks([int(i * self.y_mod / 5) for i in range(6)], y_labels)

        self.ax.imshow(self.im, zorder=0, extent=[0.1, self.x_mod, 0.1, self.y_mod])

        self.ax.set_xlim([0, self.x_mod])
        self.ax.set_ylim([0, self.y_mod])
        [spine.set_visible(False) for spine in self.ax.spines.values()]  # remove chart outlines

    def get_location_rectangles(self, to_plot_locations=None):
        """ Draw the rectangles used to identify locations. """
        self.ax.imshow(self.im, zorder=0, extent=[0, self.x_mod, 0, self.y_mod])

        for name, coord in LOCATIONS.items():
            x_west, y_north = scale_to_img((coord[0], coord[1]), (self.x_mod, self.y_mod))
            x_east, y_south = scale_to_img((coord[2], coord[3]), (self.x_mod, self.y_mod))
            self.ax.add_patch(
                plt.Rectangle((x_west, y_south), x_east - x_west, y_north - y_south, edgecolor='red', facecolor='none'))
            plt.text(x_west, (y_north + y_south) / 2, name if name not in conf.loc_eng else conf.loc_eng[name])
        if to_plot_locations is not None:
            for lat, lon in to_plot_locations:
                x, y = scale_to_img((lat, lon), (self.x_mod, self.y_mod))
                self.ax.scatter(x, y)

        boundary = (51.2529, 4.3842, 51.2271, 4.4539)
        x_labels = [f"{boundary[1] + i * (boundary[3] - boundary[1]) / 5:5.3f}" for i in range(6)]
        self.ax.set_xticks([int(i * self.x_mod / 5) for i in range(6)], x_labels)

        y_labels = [f"{boundary[2] + i * (boundary[0] - boundary[2]) / 5:5.3f}" for i in range(6)]
        self.ax.set_yticks([int(i * self.y_mod / 5) for i in range(6)], y_labels)

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "rectangles.png"))
        plt.show()

    def update(self, i):
        """ Draw figure for a single timestamp. """
        while self.ax.patches:
            self.ax.patches.pop()
        dt = pd.to_datetime(self.times[i], unit="s")
        self.ax.set_title(f'{dt.hour:2d}:{dt.minute:2d}:{dt.second:2d}')
        print(f"Frame {i}")

        # Show each Vessel at this timestamp.
        for node in self.time_to_vs[self.times[i]]:
            vessel_state = node.vs
            course = vessel_state.course
            y, x = scale_to_img((vessel_state.lat, vessel_state.lon), (self.x_mod, self.y_mod))

            vessel = node.vt.vessel
            color = (0.5, 0.5, 0.5)
            if not math.isnan(vessel_state.speed):
                color = plasma.colors[min(255, int(40 * vessel_state.speed))]

            # Draw the vessel
            patch = self.ax.add_patch(get_ship_patch((y, x), vessel.length_max, vessel.width_max, 90 - course, color))
            self.active_patches.append(patch)
            # Add an extra mark for ships in a convoy
            if conf.draw_convoy and vessel.v_id in [v for v_list in self.convoy_timings.get(self.times[i], list()) for v
                                                    in v_list]:
                plt.scatter(y, x, color="red")

        # Add a line between convoys
        if conf.draw_convoy and self.times[i] in self.convoy_timings.keys():
            for tup in self.convoy_timings[self.times[i]]:
                nodes = [n for n in self.time_to_vs[self.times[i]] if n.vs.v_id in tup]
                if len(nodes) < 2:
                    print("There should be 2 nodes", nodes)
                    continue
                y, x = scale_to_img((nodes[0].vs.lat, nodes[0].vs.lon), (self.x_mod, self.y_mod))
                y2, x2 = scale_to_img((nodes[1].vs.lat, nodes[1].vs.lon), (self.x_mod, self.y_mod))
                plt.plot([y, y2], [x, x2], color="orange")
