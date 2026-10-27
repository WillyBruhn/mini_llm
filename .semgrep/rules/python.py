import glob
import logging
import os
import sys
from pathlib import Path

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


# --- prefer-pathlib ---
def uses_os_for_paths() -> None:
    # ruleid: prefer-pathlib
    os.path.join("a", "b")
    # ruleid: prefer-pathlib
    os.path.exists("a")
    # ruleid: prefer-pathlib
    os.path.dirname(__file__)
    # ruleid: prefer-pathlib
    os.makedirs("a", exist_ok=True)
    # ruleid: prefer-pathlib
    os.listdir("a")
    # ruleid: prefer-pathlib
    os.getcwd()
    # ruleid: prefer-pathlib
    glob.glob("*.py")
    # ok: prefer-pathlib
    os.environ.get("HOME")
    # ok: prefer-pathlib
    os.fsencode("a")
    # ok: prefer-pathlib
    Path("a") / "b"


# --- no-relative-import ---
# ruleid: no-relative-import
from . import sibling
# ruleid: no-relative-import
from .sibling import name
# ruleid: no-relative-import
from ..parent import name
# ruleid: no-relative-import
from .sibling import (
    name_a,
    name_b,
)
# ok: no-relative-import
from mini_llm.sibling import name

# ruleid: no-relative-import
module = import_module(".sibling", __name__)
# ok: no-relative-import
module = import_module("mini_llm.sibling")

# --- no-import-star ---
# ruleid: no-import-star
from mini_llm.sibling import *
# ok: no-import-star
from mini_llm.sibling import name

# --- no-sys-path-manipulation ---
# ruleid: no-sys-path-manipulation
sys.path.insert(0, "/some/dir")
# ruleid: no-sys-path-manipulation
sys.path.append("/some/dir")
# ruleid: no-sys-path-manipulation
sys.path = ["/some/dir"]
# ruleid: no-sys-path-manipulation
sys.path += ["/some/dir"]
# ok: no-sys-path-manipulation
print_path = list(sys.path)
