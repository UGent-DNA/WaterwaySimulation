### Get initial XML
Run the following request in [Overpas Turbo](https://www.overpass-turbo.eu) and copy the xml-result:

[out:xml][timeout:100][bbox:51.2,4.2,51.32,4.6];
( node["waterway"];
  way["waterway"];
  relation["waterway"]; );
out geom;

### Convert to SUMO network XML with shipping options
Next run netconvert in the terminal with the following options:

netconvert --osm-files waterway_net.osm.xml -o waterways.xml 
-t $SUMO_HOME/data/typemap/osmNetconvert.typ.xml,$SUMO_HOME/data/typemap/osmNetconvertShips.typ.xml 
--geometry.remove --ramps.guess --junctions.join
(extra options for traffic lights are not necessary) --tls.guess-signals --tls.discard-simple --tls.join --tls.default-type actuated



1) Download belgium data from download.geofabrik.de and extract to resources
2) [old] osmosis --read-xml belgium-latest.osm --bounding-box top=51.2635 left=4.3772 bottom=51.2292 right=4.4832 --tf accept-nodes waterways=* --tf accept-ways waterways=* --tf accept-relations waterways=* --write-xml file=port_antwerp_small.osm
3) osmosis --read-xml belgium-latest.osm --bounding-box top=51.2635 left=4.3772 bottom=51.2292 right=4.4832 --write-xml file=port_antwerp_small.osm
4) osmium tags-filter port_antwerp_small.osm natural=water waterway=riverbank highway -o port_waterways.osm
5) netconvert --osm-files belgium_waterways.osm -o belgium_waterways.xml -t $SUMO_HOME/data/typemap/osmNetconvert.typ.xml,$SUMO_HOME/data/typemap/osmNetconvertShips.typ.xml,$SUMO_HOME/data/typemap/osmNetconvertShipAmenities.typ.xml --geometry.remove --ramps.guess --junctions.join


