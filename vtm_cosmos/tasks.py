"""Task names describe which actions are supplied; temporal_mode controls attention.

action_causal: history actions condition joint prediction of future images/actions.
action_full: supplied sensor/view actions condition future image prediction;
the causal prefix strategy restricts their visibility to the current/past times.
"""

TASK_MODES = ("action_causal", "forward_dynamics", "action_full")
_ALIASES = {"joint": "action_causal", "action_joint": "action_full", "action_casual": "action_causal"}


def canonical_task_mode(value: str) -> str:
    mode = _ALIASES.get(str(value), str(value))
    if mode not in TASK_MODES:
        raise ValueError(f"task_mode must be one of {TASK_MODES}, got {value}")
    return mode


def normalize_task_config(cfg):
    """Canonicalize in-memory configs before branching or saving; leave checkpoint files intact."""
    cfg.model.task_mode = canonical_task_mode(cfg.model.get("task_mode", "action_causal"))
    return cfg


def prediction_contract(cfg):
    """Behavioral checkpoint contract; omitted keys preserve historical behavior."""
    from vtm_cosmos.model.topology import normalize_connections

    m = cfg.get("model", {})
    training = cfg.get("training", {}) or {}
    inference = cfg.get("inference", {}) or {}
    strategy = training.get("prediction_strategy", "joint")
    contract = (canonical_task_mode(m.get("task_mode", "action_causal")),
            m.get("temporal_mode", "causal"), m.get("attention_topology", "legacy"), strategy,
            inference.get("prediction_strategy", "causal_autoregressive" if strategy == "causal_teacher_forcing" else "joint"))
    connections = normalize_connections(contract[0], contract[2], m.get("attention_connections"))
    # Preserve historical contracts; custom routing is additionally part of the
    # checkpoint identity, independent of YAML row/key ordering.
    if connections is not None:
        contract = contract + (tuple(connections.items()),)
    active = m.get("active_streams")
    if active:  # a stream-subset run (e.g. main view + view actions only) is its own contract
        from vtm_cosmos.streams import normalize_active_streams
        contract = contract + (("active_streams",) + normalize_active_streams(contract[0], tuple(active)),)
    return contract


def validate_prediction_config(cfg):
    task, temporal, topology, strategy, inference = prediction_contract(cfg)[:5]
    if temporal not in ("causal", "chunk_bidirectional"):
        raise ValueError(f"Unknown temporal_mode: {temporal}")
    if strategy == "causal_teacher_forcing":
        if (task, temporal, inference) != ("action_full", "causal", "causal_autoregressive"):
            raise ValueError("Causal prediction requires action_full / causal / causal_autoregressive")
        if float((cfg.get("training", {}).get("wrist_dropout", {}) or {}).get("probability", 0)) != 0:
            raise ValueError("The causal reference requires wrist_dropout.probability=0 for reproducible prefix training")
    elif strategy != "joint" or inference != "joint" or topology not in ("legacy", "custom"):
        raise ValueError("Unsupported prediction strategy/topology combination")
    elif task == "action_full" and temporal != "chunk_bidirectional":
        raise ValueError("Causal action_full requires training.prediction_strategy=causal_teacher_forcing")
    return cfg
