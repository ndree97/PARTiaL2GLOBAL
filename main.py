"""
PARTiaL2GLOBAL Unified Application Entry Point
==============================================
Launches the PyQt5 Graphical User Interface by default.
If CLI arguments (e.g., -s / --source) are provided, switches to CLI mode.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def main():
    # If source flag or help is provided, route to CLI
    cli_flags = {"-s", "--source", "-t", "--target", "-h", "--help", "--cli"}
    if any(arg in cli_flags for arg in sys.argv[1:]):
        from cli import main as cli_main
        cli_main()
    else:
        from partial2global.gui.app_window import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
