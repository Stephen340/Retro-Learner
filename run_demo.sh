#!/bin/bash
set -e

# Install the required Python packages
python -m pip install -r ./requirements.txt --no-cache-dir

# Import the Super Mario Bros ROM
python -m retro.import 'SMBRom'

# Import the Sonic ROM
python -m retro.import 'SonicRom'

# Run the Mario Demo script
python ./Mario_Demo.py
