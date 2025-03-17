import asyncio
import argparse
import sys
from typing import Dict, Any, Optional

from .core.logging import get_logger, configure_logging
from .core.config import ConfigManager
from .ui.cli import main_cli
from .ui.gradio_app import launch_ui

logger = get_logger("main")

async def main() -> int:
    """
    Main application entry point.
    
        int: Exit code
    """
    parser = argparse.ArgumentParser(description="AI Agent Workflow Framework")
    
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--cli", action="store_true", help="Launch CLI")
    
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", help="Log file path")
    
    parser.add_argument("--port", type=int, default=7860, help="Web interface port")
    parser.add_argument("--host", default="0.0.0.0", help="Web interface host")
    parser.add_argument("--share", action="store_true", help="Create a publicly shareable link")
    
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.debug else "INFO"
    configure_logging(log_level, args.log_file)
    
    config = ConfigManager(args.config)
    
    try:
        if args.web:
            logger.info("Launching web interface")
            launch_ui(
                share=args.share,
                server_port=args.port,
                server_name=args.host
            )
            return 0
        elif args.cli:
            logger.info("Launching CLI")
            return await main_cli()
        else:
            logger.info("No interface specified, launching web interface")
            launch_ui(
                share=args.share,
                server_port=args.port,
                server_name=args.host
            )
            return 0
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))