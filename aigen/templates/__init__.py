"""Templates for code generation."""


To run the application, create a `run.py` file in the root directory:

"""Run the AI Generator application."""

import logging
import argparse
from aigen import GradioApp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the application."""
    parser = argparse.ArgumentParser(description="Run the AI Generator application")
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1", 
        help="Hostname to run the server on"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the server on"
    )
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Whether to create a public link"
    )
    args = parser.parse_args()
    
    app = GradioApp(
        title="AI Generator Framework",
        description="Create and manage custom AI agents"
    )
    
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()
