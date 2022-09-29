# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class ExternalPropertyFileReferences(object):
    """References to external property files that should be inlined with the content of a root log file."""

    addresses: Any
    artifacts: Any
    conversion: Any
    driver: Any
    extensions: Any
    externalized_properties: Any
    graphs: Any
    invocations: Any
    logical_locations: Any
    policies: Any
    properties: Any
    results: Any
    taxonomies: Any
    thread_flow_locations: Any
    translations: Any
    web_requests: Any
    web_responses: Any


# flake8: noqa
