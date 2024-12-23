import os
import logging
from functools import lru_cache


class LoggerConfig:
    def __init__(
        self,
        log_dir: str = "logs",
        log_file_name: str = "log.log",
        log_level: int = logging.INFO,
    ):
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.log_level = log_level

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.file_formatter = logging.Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.console_formatter = logging.Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def _create_handlers(self):
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, self.log_file_name)
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.console_formatter)

        return [file_handler, console_handler]

    @lru_cache
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        logger = logging.getLogger(name)

        # Only add handlers if the logger doesn't have any
        if not logger.handlers:
            logger.setLevel(self.log_level)
            for handler in self._create_handlers():
                logger.addHandler(handler)

        return logger


default_config = LoggerConfig()


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger using the default configuration."""
    if name is None:
        # Use the caller's module name if none provided
        import inspect

        frame = inspect.stack()[1]
        name = frame.frame.f_globals["__name__"]
    return default_config.get_logger(name)
