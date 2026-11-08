"""Validated stream routing, independent of the temporal attention constraint."""

from collections.abc import Mapping, Sequence


ATTENTION_TOPOLOGIES = ("legacy", "all_streams_causal", "custom")


def normalize_connections(task_mode, topology, connections=None):
    """Return canonical query -> key tuples; never add reverse or self edges.

    Explicit self edges guarantee a valid key at the earliest timestep, even
    when optional wrist streams are absent. Every task stream must have a row.
    """
    from vtm_cosmos.streams import ALL_STREAMS, STREAMS
    from vtm_cosmos.tasks import canonical_task_mode

    if topology not in ATTENTION_TOPOLOGIES:
        raise ValueError(f"Unknown attention topology: {topology}")
    if topology != "custom":
        if connections is not None:
            raise ValueError("attention_connections requires attention_topology: custom")
        return None
    streams = ALL_STREAMS if canonical_task_mode(task_mode) == "action_full" else STREAMS
    if not isinstance(connections, Mapping):
        raise ValueError("custom topology requires attention_connections: query stream -> list of key streams")
    missing = set(streams) - set(connections)
    unknown = set(connections) - set(streams)
    if missing or unknown:
        raise ValueError(f"attention_connections rows: missing={sorted(missing)}, unknown={sorted(unknown)}")
    result = {}
    for query in streams:
        keys = connections[query]
        if isinstance(keys, str) or not isinstance(keys, Sequence) or any(not isinstance(k, str) for k in keys):
            raise ValueError(f"attention_connections.{query} must be a list of stream names")
        unknown = set(keys) - set(streams)
        if unknown:
            raise ValueError(f"attention_connections.{query}: unknown key streams {sorted(unknown)}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"attention_connections.{query}: duplicate key streams")
        if query not in keys:
            raise ValueError(f"attention_connections.{query} must include itself to avoid empty causal attention rows")
        result[query] = tuple(s for s in streams if s in keys)
    return result
