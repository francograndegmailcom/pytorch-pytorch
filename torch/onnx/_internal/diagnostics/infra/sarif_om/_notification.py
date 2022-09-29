# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class Notification(object):
    """Describes a condition relevant to the tool itself, as opposed to being relevant to a target being analyzed by the tool."""

    message: Any
    associated_rule: Any
    descriptor: Any
    exception: Any
    level: Any
    locations: Any
    properties: Any
    thread_id: Any
    time_utc: Any


# flake8: noqa
