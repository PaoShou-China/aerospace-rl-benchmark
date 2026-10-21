"""Shared success-rate-driven difficulty curriculum for all task envs.

Project-wide curriculum contract: every env exposes a difficulty in [0, 1]
(0 easiest, 1 hardest = the published benchmark for every task except quadx,
whose deterministic-spawn benchmark is the easiest variant and whose axis
extends toward harder spawn perturbations). A static difficulty is a plain
constructor kwarg (`difficulty=d`); `curriculum=True` ramps the difficulty
automatically: whenever a fixed window of completed episodes closes with a
rolling success rate at or above `threshold`, the difficulty advances by
`delta` (forward-only, the docking precedent).

Envs own the difficulty->knob mapping (`_apply_difficulty`); this tracker
only owns the statistics and the advance gate, so every task shares identical
progression semantics. Counters are CPU ints -- update() is called at most
once per env step when an episode boundary occurred, so the sync cost is
negligible.
"""


class SuccessRateCurriculum:
    """Windowed success-rate gate advancing difficulty in fixed steps."""

    def __init__(self,start=0.,window=1024,threshold=.6,delta=.1):
        if not 0.<=start<=1.:raise ValueError('start must be in [0,1]')
        if window<1:raise ValueError('window must be positive')
        if not 0.<=threshold<=1.:raise ValueError('threshold must be in [0,1]')
        if delta<=0.:raise ValueError('delta must be positive')
        self.difficulty=float(start)
        self.window=int(window)
        self.threshold=float(threshold)
        self.delta=float(delta)
        self.eps=0                # completed episodes in the current window
        self.succ=0               # successes among them
        self.last_rate=None       # rolling success rate at the last window close

    def update(self,done,complete):
        """Accumulate one step's episode boundaries; advance at window close.

        done/complete: the env's reset/complete boolean masks (device tensors).
        Returns the new difficulty when the window closed AND advanced, else
        None. Counters reset at every window close regardless of the gate.
        """
        self.eps+=int(done.sum())
        self.succ+=int((done&complete).sum())
        if self.eps<self.window:return None
        self.last_rate=self.succ/self.eps
        self.eps=0;self.succ=0
        if self.last_rate<self.threshold or self.difficulty>=1.:return None
        # round the accumulated step so 0.1 increments read back as 0.1/0.3/...
        # instead of binary-float residue in cfg/run.json
        self.difficulty=min(1.,round(self.difficulty+self.delta,10))
        return self.difficulty
