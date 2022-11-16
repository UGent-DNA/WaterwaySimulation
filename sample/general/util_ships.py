from typing import List, Tuple

from sample.data_processing.data_classes import Vessel, VesselTrip


def get_ship_type(vessel: Vessel):
    goods = {"Vrachtschepen", "Containerschepen", "Duwvaart", "Stortgoedschepen", "Stukgoedschepen", "Tankers"}
    if vessel.ship_type in goods:
        return "Cargo"
    elif "Toerisme" in vessel.ship_type or "pleziervaart" in vessel.ship_type:
        return "Tourism"
    else:
        return "Divers"


def rename_location(loc: str):
    if loc in {"Albertkanaal", "Albertdok", "Amerikadok"}:
        return loc
    if loc in {"Suezdok", "Asiadok"}:
        return "Stadshaven"
    else:
        return "Diverse"


def get_source_destination(vessel_trip: VesselTrip) -> Tuple[str, str]:
    t = vessel_trip.trip
    start = vessel_trip.find_location(t[0].lat, t[0].lon)
    end = vessel_trip.find_location(t[-1].lat, t[-1].lon)
    return rename_location(start), rename_location(end)


def detect_intersection_as_turning_bowl(vts: List[VesselTrip]):
    for vt in vts:
        trip_path = vt.get_path()

        intersection_duration = sum(dur for loc, dur in trip_path if loc == "Intersection")
        if trip_path[0][0] == trip_path[-1][0] == "Amerikadok" and len(trip_path) < 5 and intersection_duration > 100:
            print(vt.trip[0].time, trip_path, vt.vessel)
