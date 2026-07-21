"""Input/output: loading flights and maps."""
from .flight import FlightData, load_flight, segment_indices, inspect_h5
from .maps import MagMap, load_map

__all__ = ["FlightData", "load_flight", "segment_indices", "inspect_h5", "MagMap", "load_map"]
