import logging

logger = logging.getLogger(__name__)


# --- no-print ---
def uses_print() -> None:
    # ruleid: no-print
    print("hello")
    # ruleid: no-print
    print("no newline", end="", flush=True)
    # ok: no-print
    logger.info("hello")


# --- no-import-inside-function ---
def imports_in_function() -> None:
    # ruleid: no-import-inside-function
    import json

    # ruleid: no-import-inside-function
    import numpy as np

    # ruleid: no-import-inside-function
    from collections import OrderedDict

    return json, np, OrderedDict


async def imports_in_async_function() -> None:
    # ruleid: no-import-inside-function
    import sys

    return sys


class Example:
    def method(self) -> None:
        # ruleid: no-import-inside-function
        from typing import Any

        return Any


def outer() -> None:
    def inner() -> None:
        # ruleid: no-import-inside-function
        import re

        return re

    return inner


# ok: no-import-inside-function
import json  # noqa: E402
