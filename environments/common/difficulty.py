"""Static difficulty-axis registry for the env side.

Every task env accepts ``difficulty`` in [0, 1] (``None`` = benchmark
behavior, matching each env's ``benchmark_difficulty=1.`` contract) and
every self-contained train entry exposes a hard-coded ``DIFFICULTY``
constant.  This module is the single point that maps the (task, difficulty)
pair to env kwargs, so the convention cannot drift:

    from environments.common.difficulty import resolve_env_kwargs

    DIFFICULTY = None
    ENV_KWARGS = resolve_env_kwargs('docking', DIFFICULTY)

Adding a task: append its id to ``TASK_IDS`` (one line).
"""

TASK_IDS = ('quadx', 'fixedwing', 'rocket', 'satellite',
            'transfer', 'docking', 'rendezvous')


def resolve_env_kwargs(task_id, difficulty):
    """Validate (task_id, difficulty) and return the env kwargs dict.

    Returns ``{}`` for the benchmark configuration (``difficulty is None``)
    and ``{'difficulty': d}`` otherwise; raises on unknown tasks or out-of-
    range difficulties so a typo fails fast at import time of the entry.
    """
    if task_id not in TASK_IDS:
        raise KeyError(f'unknown task id {task_id!r}; known: {TASK_IDS}')
    if difficulty is None:
        return {}
    if not 0.0 <= float(difficulty) <= 1.0:
        raise ValueError(f'{task_id}: difficulty must be in [0, 1], '
                         f'got {difficulty!r}')
    return {'difficulty': float(difficulty)}
