"""Version B: NO interference, with the global weakest fraction changed to 30%.

Run: python -m mira_leo_study.csi_elevation_with_interference
The historical module filename is retained for the existing command. Geometry,
physical CSI, expected-gain curve and plot sampling exactly match version A.
Only the global bottom fraction is 30% instead of 20%. Results are stored in
no_interference_bottom30/ so they do not overwrite either earlier version.
"""
from __future__ import annotations

from .csi_elevation_no_interference import main as common_main


def main(argv=None) -> int:
    return common_main(argv, with_interference=False, bottom_percent=30)


if __name__ == "__main__":
    raise SystemExit(main())
