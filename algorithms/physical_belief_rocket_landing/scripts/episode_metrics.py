"""Fixed-first-episode accounting extracted unchanged from the experiment evaluator."""
import math


def wilson_interval(successes, total, z=1.959963984540054):
    """Two-sided 95% Wilson score interval (not a training-seed interval)."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError('expected 0 <= successes <= total and total >= 1')
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - half), min(1., center + half)]


class FirstEpisodes:
    """Pure bookkeeping: repeated completions of a replica are ignored."""

    def __init__(self, num_envs):
        if num_envs < 1:
            raise ValueError('num_envs must be positive')
        self.records = [None] * num_envs

    def add(self, replica, record):
        if not 0 <= replica < len(self.records):
            raise IndexError(replica)
        if self.records[replica] is not None:
            return False
        self.records[replica] = {'replica': replica, **record}
        return True

    def summary(self):
        completed = [r for r in self.records if r is not None]
        n = len(self.records)
        successes = sum(bool(r['official_success']) for r in completed)
        interval = wilson_interval(successes, n)
        settled = sum(bool(r.get('settled_now')) for r in completed)
        reasons = {}
        for r in completed:
            if not r['official_success']:
                reason = r['failure_reason']
                reasons[reason] = reasons.get(reason, 0) + 1
        missing = n - len(completed)
        if missing:
            reasons['incomplete_at_full_horizon'] = missing
        return {
            'denominator': n, 'episodes_finished': len(completed),
            'incomplete_episodes': missing, 'successes': successes,
            'success_rate': successes / n, 'wilson_95_interval': interval,
            'point_estimate_at_least_99_percent': successes / n >= .99,
            'wilson_lower_at_least_99_percent': interval[0] >= .99,
            'complete_evaluation': missing == 0,
            'failure_counts': reasons,
            'final_settled_successes': settled,
            'final_settled_rate': settled / n,
            'final_settled_wilson_95_interval': wilson_interval(settled,n),
            'success_then_failure': sum(bool(r.get('success_then_failure')) for r in completed),
        }
