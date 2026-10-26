"""Training-only successful-state curriculum; thresholds are local choices."""
import torch


STATE_WIDTH = 31  # p3, q4, v3, w3, throttle4, pwm4, PID integral3/previous3, action4


class InitialStateBuffer:
    def __init__(self, device, capacity=4096, min_states=128, min_episodes=64,
                 success_threshold=.6, replay_fraction=.5):
        if capacity < 1 or not 1 <= min_states <= capacity or min_episodes < 1:
            raise ValueError('invalid buffer size or curriculum warmup')
        if not 0 <= success_threshold <= 1 or not 0 <= replay_fraction <= 1:
            raise ValueError('curriculum probabilities must be in [0,1]')
        self.states=torch.zeros(capacity,STATE_WIDTH,device=device,dtype=torch.float64)
        self.gates=torch.zeros(capacity,device=device,dtype=torch.long)
        self.min_states,self.min_episodes=min_states,min_episodes
        self.success_threshold,self.replay_fraction=success_threshold,replay_fraction
        self.count,self.cursor,self.episodes,self.success_ema=0,0,0,0.

    @property
    def ready(self):
        return (self.count>=self.min_states and self.episodes>=self.min_episodes
                and self.success_ema>=self.success_threshold)

    def add(self, states, gates, mask):
        valid=mask & torch.isfinite(states).all(-1)
        selected=states[valid].detach()[-len(self.states):]
        gates=gates[valid].detach()[-len(self.states):]
        n=len(selected)
        ids=(torch.arange(n,device=states.device)+self.cursor)%len(self.states)
        self.states[ids],self.gates[ids]=selected,gates
        self.cursor=(self.cursor+n)%len(self.states)
        self.count=min(len(self.states),self.count+n)

    def update(self, done, success):
        n=int(done.sum())
        if n:
            rate=float(success[done].to(torch.float64).mean())
            self.success_ema=rate if not self.episodes else .95*self.success_ema+.05*rate
            self.episodes+=n

    def sample(self, rng, mask):
        draws=rng.uniform(2,mask)
        ids=(draws[:,0]*max(self.count,1)).long()
        use=mask & (draws[:,1]<self.replay_fraction) & self.ready
        return self.states[ids],self.gates[ids],use

    def state_dict(self):
        return dict(states=self.states[:self.count].cpu().clone(),gates=self.gates[:self.count].cpu().clone(),
                    count=self.count,cursor=self.cursor,episodes=self.episodes,success_ema=self.success_ema)

    def load_state_dict(self, state):
        count=int(state['count'])
        if count>len(self.states) or state['states'].shape!=(count,STATE_WIDTH):
            raise ValueError('incompatible initialization buffer checkpoint')
        self.states[:count].copy_(state['states']);self.gates[:count].copy_(state['gates'])
        self.count,self.cursor=count,int(state['cursor'])
        self.episodes,self.success_ema=int(state['episodes']),float(state['success_ema'])
