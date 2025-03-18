# Dynamic agent modules

import sys
import os

# Add the project root to the Python path to make absolute imports work
# This helps resolve imports like "from aigen.tools.factory import create_tool"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
