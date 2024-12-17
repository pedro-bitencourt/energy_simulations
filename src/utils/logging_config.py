# src/logging_config.py
# Sets up the logging configuration for the project

import logging
from typing import Optional

from rich.logging import RichHandler

def setup_logging(level: Optional[int] = logging.DEBUG) -> None:
    """
    Configure logging with Rich handler and set up basic configuration.
    
    Args:
        level: The logging level to use. Defaults to logging.DEBUG.
    """
    # Configure the handler with pretty printing enabled
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=True,
        markup=True
    )
    
    # Configure basic logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler]
    )
    
    # Ensure root logger uses the rich handler
    logging.getLogger().handlers = [rich_handler]
    
    # Suppress matplotlib debug messages
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


