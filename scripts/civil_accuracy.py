"""Continuous first-episode accuracy measures derived from outgoing observations."""
import torch


class CivilAccuracy:
    def __init__(self, task, env):
        self.task = task
        self.values = {}
        self.count = torch.zeros(env.num_envs, device=env.device)
        self.visible = torch.zeros_like(self.count)
        self.error_sum = torch.zeros_like(self.count)
        self.final_waypoint = (
            env.targets[:, -1].clone() if task in ("quadx", "fixedwing", "quadplane") else None
        )
        self.ref_offset = env.ref_offset if task == "quadx_track" else None

    def update(self, obs, active, done):
        self.count += active
        values = {}
        if self.final_waypoint is not None:
            values["terminal_distance_to_final_waypoint_m"] = (
                obs[:, 10:13] - self.final_waypoint
            ).norm(dim=-1)
        elif self.task in ("docking", "rendezvous"):
            values["terminal_relative_distance_m"] = obs[:, :3].norm(dim=-1)
            values["terminal_relative_speed_m_s"] = obs[:, 3:6].norm(dim=-1)
        elif self.task == "satellite":
            values["terminal_attitude_error_rad"] = obs[:, 4:7].norm(dim=-1)
            values["terminal_spin_rad_s"] = obs[:, 7:10].norm(dim=-1)
        elif self.task == "quadx_track":
            visible = active & (obs[:, -1] > .5)
            err = (obs[:, -4:-2] + self.ref_offset).norm(dim=-1)
            self.visible += visible
            self.error_sum += torch.where(visible, err, 0.)
        for name, value in values.items():
            if name not in self.values:
                self.values[name] = torch.zeros_like(value)
            self.values[name].copy_(torch.where(done & active, value, self.values[name]))

    def attach(self, book, rows):
        metrics = dict(self.values)
        metrics["policy_steps"] = self.count
        if self.task == "quadx_track":
            metrics["camera_visible_fraction"] = self.visible / self.count.clamp_min(1)
            metrics["visible_measured_offset_error_m"] = self.error_sum / self.visible.clamp_min(1)
        if self.task == "transfer":
            for key in ("perigee_raise_km", "apo_drift_km", "propellant_kg"):
                metrics[key] = book.metrics[key]
        for name, tensor in metrics.items():
            for row, value in zip(rows, tensor.cpu().tolist(), strict=True):
                row[name] = value
        if self.task == "quadx_track":
            for row in rows:
                if row["camera_visible_fraction"] == 0:
                    row["visible_measured_offset_error_m"] = None
