"""Cache-free entry point; no backend import until an explicitly started stage."""
import sys
sys.dont_write_bytecode = True
from point_select.cli import main

if __name__ == "__main__":
    main()
