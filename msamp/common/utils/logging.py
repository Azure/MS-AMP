# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP logging module."""

import socket
import logging
import sys

import colorlog
# workaround to get rid of isatty from
# colorama StreamWrapper in WSL2
try:
    from colorama import deinit
    deinit()
except Exception:
    pass


class LoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter class which add customized function for log error and raise exception."""
    def log_and_raise(self, exception, msg, *args):
        """Log error and raise exception.

        Args:
            exception (BaseException): Exception class.
            msg (str): logging message.
            args (dict): arguments dict for message.
        """
        self.error(msg, *args)
        raise exception(msg % args)


class MsAmpLogger:
    """MsAMP Logger class."""
    @staticmethod
    def add_handler(logger, stream=sys.stdout, filename=None, color=False):
        """Add handler for logger.

        Args:
            logger (Logger): Logger to which the handler is added.
            stream (IO, optional): The stream that the stream handler should use. Defaults to sys.stdout.
            filename (str, optional): The filename that file handler should use. Defaults to None.
            color (bool, optional): Colored format or not. Defaults to False.
        """
        formatter = logging.Formatter(
            '[%(asctime)s %(hostname)s:%(process)d][%(filename)s:%(lineno)s][%(levelname)s] %(message)s'
        )
        if color:
            formatter = colorlog.ColoredFormatter(
                '%(reset)s'
                '[%(cyan)s%(asctime)s %(hostname)s:%(process)d%(reset)s]'
                '[%(blue)s%(filename)s:%(lineno)s%(reset)s]'
                '[%(log_color)s%(levelname)s%(reset)s] %(message)s'
            )

        handler = logging.NullHandler()
        if filename:
            # Create file handler if filename exists
            handler = logging.FileHandler(filename)
        else:
            # Create stream handler otherwise
            handler = logging.StreamHandler(stream=stream)

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @staticmethod
    def create_logger(name, level=logging.INFO):
        """Create logger instance with customized format.

        Args:
            name (str): Logger name.
            level (int): Logging level. Defaults to logging.INFO.

        Return:
            Logger: logger with the specified name, level and adapters.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        MsAmpLogger.add_handler(logger, stream=sys.stdout, color=True)
        logger = LoggerAdapter(logger, extra={'hostname': socket.gethostname()})

        return logger


logger = MsAmpLogger.create_logger('msamp')
