"""Main CLI entrypoint for Project Robbie."""

import sys
import argparse
from robbie import __version__
from robbie.config import RobbieConfig
from robbie.utils.logger import logger
from robbie.utils.device import get_hardware_info

def parse_args():
    parser = argparse.ArgumentParser(description="Project Robbie CLI")
    parser.add_argument("--version", action="version", version=f"Project Robbie v{__version__}")
    parser.add_argument("--info", action="store_true", help="Print system and hardware report")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting Project Robbie (v{__version__})")

    if args.info:
        report = get_hardware_info()
        print("\n=== Project Robbie Hardware Report ===")
        print(f"Device Type:     {report.device_type}")
        print(f"Device Name:     {report.device_name}")
        print(f"CUDA Available:  {report.cuda_available}")
        print(f"GPU Count:       {report.device_count}")
        if report.cuda_available:
            print(f"Total VRAM:      {report.vram_total_gb} GB")
            print(f"Available VRAM:  {report.vram_free_gb} GB")
        print("======================================\n")

if __name__ == "__main__":
    main()
