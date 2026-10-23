"""Run bounded counterfactual feedback search, distillation, and real evaluation.

Example:
  .venv/bin/python algorithms/moe_rocket_landing/train.py \
      --out results/rocket_feedback_repair

All physical rollouts start at natural reset in an immutable upstream snapshot.
Search replicas share prefixes and paired future disturbance streams.  Every
1200-step branch, including its prefix and later auto-reset episodes, is billed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import runtime as baseline

HORIZON = 1200
TRAIN_SEED = 710007
NEIGHBOR_SEED = 720007
STUDENT_SEED = 730007
TRAIN_CAP = 1_920_000
BACKTRACK = (16, 64, 256, 512)


def emit(event, **data):
    print(json.dumps({"event": event, **data}, allow_nan=False), flush=True)


def paired_summary(parent, candidate):
    if len(parent) != len(candidate):
        raise ValueError("pair lengths differ")
    gained = lost = 0
    for a, b in zip(parent, candidate):
        if a["initial_state_p_q_v_w"] != b["initial_state_p_q_v_w"]:
            raise RuntimeError("paired initial physical states differ")
        gained += int(b["official_success"] and not a["official_success"])
        lost += int(a["official_success"] and not b["official_success"])
    return {"denominator": len(parent),
            "parent_successes": sum(int(r["official_success"]) for r in parent),
            "candidate_successes": sum(int(r["official_success"]) for r in candidate),
            "gained": gained, "lost": lost, "net": gained - lost,
            "all_initial_states_identical": True}


class Experiment:
    def __init__(self, args):
        self.args, self.out = args, args.out.resolve()
        self.started = time.perf_counter()
        if not args.parent.is_file():
            raise FileNotFoundError(args.parent)
        self.source = baseline.prepare_snapshot(self.out)
        self.parent_dir = baseline.HERE / "evidence/parent"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(args.torch_threads)
        os.environ["QD_NUM_THREADS"] = str(args.physics_threads)
        self.torch, _ = baseline.setup_imports(self.source, self.out, args.torch_threads)
        payload = self.torch.load(args.parent, map_location="cpu", weights_only=False)
        parent_payload = payload.get("parent_payload", payload)
        self.torch.save(parent_payload, self.out / "parent_checkpoint.pt")
        implementation = self.out / "implementation"
        implementation.mkdir()
        for path in baseline.HERE.glob("*.py"):
            shutil.copy2(path, implementation / path.name)
        self.status = {"status": "initializing", "training_transitions": 0,
                       "evaluation_transitions": 0, "environment_transitions": 0,
                       "completed_rollouts": [], "training_cap": TRAIN_CAP}
        self.manifest = {
            "method": "phase-aware landing with gated feedback repair",
            "anchor_commit": baseline.ANCHOR, "backend": args.backend, "engine_precision": 32,
            "source_checkpoint_sha256": baseline.sha256(args.parent),
            "parent_sha256": baseline.sha256(self.out / "parent_checkpoint.pt"),
            "source_archive_sha256": baseline.sha256(self.out / "upstream_source.tar.gz"),
            "implementation_sha256": {p.name: baseline.sha256(p) for p in implementation.glob("*.py")},
            "train_seeds": {"screening": TRAIN_SEED, "neighborhood": NEIGHBOR_SEED, "on_policy": STUDENT_SEED},
            "development_seeds": baseline.DEV_SEEDS,
            "physical_dynamics_modified": False, "success_predicate_modified": False,
            "termination_modified": False, "state_restoration_used": False,
            "training_uses_development_labels": False, "sealed_test_used": False,
            "teacher_calls": 0, "max_steps": HORIZON, "training_cap": TRAIN_CAP,
            "activation_height": 150.0,
            "search": {"max_failure_cases": 16, "initial_candidates_per_case": 16,
                       "disturbances": 2, "rounds": 2, "backtrack_control_steps": BACKTRACK},
            "budget_counts": "all natural-reset prefixes and full vector advances",
            "evaluation_protocol": "upstream_source_first_episode_full_horizon_v1",
            "max_parallel_environments": args.max_envs,
        }
        baseline.write_json(self.out / "manifest.json", self.manifest)
        self.save()
        self.parent = None
        self.rollout_count = 0

    def save(self, **updates):
        self.status.update(updates)
        self.status["wall_seconds"] = time.perf_counter() - self.started
        baseline.write_json(self.out / "status.json", self.status)

    def load_parent(self, obs):
        from phase_policy import RocketPhaseModel
        from expert_router import FrozenExpertSelector, RidgeAdvantageGate
        torch = self.torch
        payload = torch.load(self.out / "parent_checkpoint.pt", map_location=obs.device, weights_only=False)
        actors = []
        for cfg, weights in zip(payload["expert_configs"], payload["expert_actor_state_dicts"]):
            cfg = copy.deepcopy(cfg)
            kwargs = cfg["actor"]
            kwargs.pop("class_name")
            std = kwargs.get("distribution_cfg", {}).get("init_std")
            if isinstance(std, list):
                kwargs["distribution_cfg"]["init_std"] = torch.tensor(std, device=obs.device)
            actor = RocketPhaseModel(obs, cfg["obs_groups"], "actor", 7, **kwargs).to(obs.device)
            actor.load_state_dict(weights, strict=True)
            actors.append(actor.eval().requires_grad_(False))
        gate = RidgeAdvantageGate().to(obs.device)
        gate.load_state_dict(payload["gate_state_dict"])
        return FrozenExpertSelector(*actors, gate).eval().requires_grad_(False)

    def rollout(self, name, streams, *, category="training", programs=None, student=None,
                student_mask=None, record_trace=False, splits=None, references=None):
        """One complete first episode per natural stream; no artificial reset."""
        torch = self.torch
        import genesis as gs
        import pyflyt_genesis.common.plant as plant_module
        from pyflyt_genesis.rocket import RocketLandingEnv
        from pyflyt_genesis.common.dynamics import euler
        from feedback_program import feedback_action, repair_features
        from metrics import FirstEpisodes

        n = len(streams)
        if n > self.args.max_envs:
            raise ValueError("requested rollout exceeds parallel environment cap")
        if category == "training" and self.status["training_transitions"] + n * HORIZON > TRAIN_CAP:
            raise RuntimeError("training budget would be exceeded")
        if (self.out / f"{name}.json").exists():
            raise FileExistsError(name)
        self.save(status=category, current_rollout=name, current_step=0)
        original_random = plant_module.TensorRandom

        class ExplicitNaturalStreams(original_random):
            def __init__(self, count, device, seed):
                super().__init__(count, device, seed)
                self.seed.copy_(torch.tensor(streams, dtype=torch.int64, device=device))

        torch.manual_seed(719007)
        plant_module.TensorRandom = ExplicitNaturalStreams
        try:
            env = RocketLandingEnv(num_envs=n, backend=self.args.backend, seed=streams[0], difficulty=None)
        finally:
            plant_module.TensorRandom = original_random
        tick = time.perf_counter()
        try:
            if gs.tc_float != torch.float32:
                raise RuntimeError("expected pristine float32 physics")
            device = env.device
            obs = env.get_observations()
            if self.parent is None:
                self.parent = self.load_parent(obs)
            self.parent.reset()
            if isinstance(student, dict):
                from repair_policy import load_repair_policy
                student = load_repair_policy(student, obs)
            if student is not None:
                student.reset()
            pending = torch.ones(n, dtype=torch.bool, device=device)
            initial = torch.cat(env.plant.state(), -1).detach().cpu()
            initial_hash = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
            book = FirstEpisodes(n)
            terminal_quality = torch.zeros(n, device=device)
            branch_features = torch.zeros((n, 1), device=device)  # allocated on first feature observation
            branch_obs = torch.zeros((n, 30), device=device)
            branch_seen = torch.zeros(n, dtype=torch.bool, device=device)
            prefix_error = 0.
            trigger_records = [None] * n
            trace_obs, trace_actions, trace_alive = [], [], []
            original_reset = env.reset_idx

            def record_then_reset(mask):
                first = mask & pending
                ids = first.nonzero(as_tuple=False).flatten().tolist()
                if ids:
                    if not bool((env.completed_episodes[first] == 1).all()):
                        raise RuntimeError("first-episode counter mismatch")
                    if not torch.equal(env.completed_successes[first], env.complete[first].long()):
                        raise RuntimeError("official success counter mismatch")
                    p, q, velocity, omega = env.plant.state()
                    tilt = euler(q)[:, :2].norm(dim=-1)
                    settled = (env.plant.pad_contact & ~env.fatal_collision &
                               (velocity.norm(dim=-1) < .02) & (omega.norm(dim=-1) < .02) & (tilt < .1))
                    quality = -(velocity.norm(dim=-1).clamp_max(100) / 2 +
                                omega.norm(dim=-1).clamp_max(10) * 2 + tilt.clamp_max(3) * 2 +
                                p[:, :2].norm(dim=-1).clamp_max(50) / 5)
                    terminal_quality[first] = quality[first].float()
                    values = {"official_success": env.complete, "fatal_collision": env.fatal_collision,
                              "out_of_bounds": env.out_of_bounds, "settled_now": settled,
                              "episode_length": env.count, "episode_return": env.returns,
                              "fuel": env.plant.fuel, "terminal_speed": velocity.norm(dim=-1),
                              "terminal_spin": omega.norm(dim=-1), "terminal_tilt": tilt,
                              "terminal_xy": p[:, :2].norm(dim=-1), "search_tiebreak": quality}
                    values = {k: v[first].detach().cpu().tolist() for k, v in values.items()}
                    for j, index in enumerate(ids):
                        record = {k: v[j] for k, v in values.items()}
                        record["success_then_failure"] = bool(record["official_success"] and
                                                              (record["fatal_collision"] or record["out_of_bounds"]))
                        record["failure_reason"] = (None if record["official_success"] else
                            "fatal_collision" if record["fatal_collision"] else
                            "out_of_bounds" if record["out_of_bounds"] else "timeout")
                        record.update(sample_id=f"iid-stream-{streams[index]}", rng_stream_id=streams[index],
                                      initial_state_p_q_v_w=initial[index].tolist(),
                                      intervention=trigger_records[index])
                        book.add(index, record)
                    pending[first] = False
                original_reset(mask)
                self.parent.reset(mask)
                if student is not None:
                    student.reset(mask)

            env.reset_idx = record_then_reset
            if programs is not None:
                parameters = torch.tensor([p["parameters"] for p in programs], device=device)
                modes = torch.tensor([p["mode"] for p in programs], dtype=torch.long, device=device)
                starts = torch.tensor([p["start"] for p in programs], dtype=torch.long, device=device)
                durations = torch.tensor([p["duration"] for p in programs], dtype=torch.long, device=device)
                enabled = torch.tensor([p.get("enabled", True) for p in programs], device=device)
                reference_indices = torch.tensor(references, dtype=torch.long, device=device)
            if splits is not None:
                split_steps = torch.tensor([s[0] for s in splits], dtype=torch.long, device=device)
                split_streams = torch.tensor([s[1] for s in splits], dtype=torch.long, device=device)
            if student_mask is not None:
                student_enabled = torch.tensor(student_mask, dtype=torch.bool, device=device)
            else:
                student_enabled = torch.ones(n, dtype=torch.bool, device=device)
            with torch.no_grad():
                for step in range(HORIZON):
                    raw = obs["policy"]
                    base_action = self.parent(obs)
                    if record_trace:
                        trace_obs.append(raw.detach().cpu().clone())
                        trace_actions.append(base_action.detach().cpu().clone())
                        trace_alive.append(pending.detach().cpu().clone())
                    action = base_action
                    if splits is not None:
                        split = (split_steps == step) & pending
                        env.plant.rng.seed[split] = split_streams[split]
                    if programs is not None:
                        enter = (starts == step) & pending
                        if bool(enter.any()):
                            rows = enter.nonzero(as_tuple=False).flatten()
                            refs = reference_indices[rows]
                            delta = float((raw[rows] - raw[refs]).abs().max())
                            prefix_error = max(prefix_error, delta)
                            if delta > 2e-5 or not torch.equal(env.plant.rng.counter[rows], env.plant.rng.counter[refs]):
                                raise RuntimeError(f"paired pre-intervention states differ: {delta}")
                            if not torch.equal(initial[rows.cpu()], initial[refs.cpu()]):
                                raise RuntimeError("paired natural initializations differ")
                            features = repair_features(raw, base_action)
                            if branch_features.shape[1] == 1:
                                branch_features = torch.zeros_like(features)
                            branch_features[rows] = features[rows]
                            branch_obs[rows] = raw[rows]
                            branch_seen[rows] = True
                        active = enabled & pending & (step >= starts) & (step < starts + durations)
                        if bool(active.any()):
                            action = base_action.clone()
                            action[active] = feedback_action(raw[active], base_action[active], parameters[active], modes[active])
                    elif student is not None:
                        action = student.apply_to_parent(raw, base_action, student_enabled & pending)
                        trigger = student.last_trigger
                        if trigger is not None:
                            for j, row in enumerate(trigger["indices"].tolist()):
                                trigger_records[row] = {"step": step, "program": int(trigger["program"][j]),
                                    "features": trigger["features"][j].cpu().tolist(),
                                    "confidence": float(trigger["confidence"][j]),
                                    "distance": float(trigger["distance"][j])}
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("non-finite action")
                    obs, _, _, _ = env.step(action)
                    self.status["environment_transitions"] += n
                    self.status[f"{category}_transitions"] += n
                    if (step + 1) % 200 == 0:
                        self.save(current_step=step + 1)
                if bool(pending.any()):
                    raise RuntimeError("incomplete first episodes at full horizon")
            result = {"name": name, "category": category, "backend": self.args.backend,
                      "engine_dtype": str(gs.tc_float), "anchor_commit": baseline.ANCHOR,
                      "protocol": "upstream_source_first_episode_full_horizon_v1",
                      "summary": book.summary(), "episodes": book.records,
                      "initial_state_sha256": initial_hash, "environment_transitions": n * HORIZON,
                      "scored_first_episode_steps": sum(r["episode_length"] for r in book.records),
                      "max_paired_prefix_observation_error": prefix_error,
                      "wall_seconds": time.perf_counter() - tick}
            artifact = {"branch_features": branch_features.cpu(), "branch_observations": branch_obs.cpu(),
                        "branch_seen": branch_seen.cpu(), "initial_state": initial}
            if record_trace:
                artifact.update(observations=torch.stack(trace_obs), parent_actions=torch.stack(trace_actions),
                                first_episode_active=torch.stack(trace_alive))
            torch.save(artifact, self.out / f"{name}.pt")
            baseline.write_json(self.out / f"{name}.json", result)
            self.status["completed_rollouts"].append({k: result[k] for k in
                                                     ("name", "summary", "environment_transitions", "wall_seconds")})
            self.save()
            emit("rollout_complete", name=name, successes=result["summary"]["successes"],
                 denominator=n, seconds=round(result["wall_seconds"], 2),
                 training_transitions=self.status["training_transitions"])
            return result, artifact
        finally:
            env.close()

    def screen(self):
        reports, traces = [], []
        for start in range(0, 256, min(128, self.args.max_envs)):
            count = min(128, self.args.max_envs, 256 - start)
            report, trace = self.rollout(f"screen-{start:03d}", list(range(TRAIN_SEED + start, TRAIN_SEED + start + count)),
                                         record_trace=True)
            reports += report["episodes"]
            traces.append(trace)
        torch = self.torch
        combined = {k: torch.cat([t[k] for t in traces], dim=1) for k in
                    ("observations", "parent_actions", "first_episode_active")}
        failed = [i for i, r in enumerate(reports) if not r["official_success"]]
        # Spread over the observed failure list without looking at development.
        selected = [failed[round(i * (len(failed) - 1) / max(1, min(16, len(failed)) - 1))]
                    for i in range(min(16, len(failed)))]
        cases = [{"index": i, "stream": reports[i]["rng_stream_id"],
                  "failure_step": reports[i]["episode_length"] - 1,
                  "initial_state": reports[i]["initial_state_p_q_v_w"]} for i in selected]
        baseline.write_json(self.out / "screening_summary.json", {
            "episodes": len(reports), "successes": len(reports) - len(failed), "failures": len(failed),
            "selected_failure_cases": cases, "selection": "uniformly spaced over training-only failure indices"})
        return reports, combined, cases

    def make_population(self, case, count, generator, previous=None):
        torch = self.torch
        from feedback_program import default_parameters, parameter_bounds
        low, high = parameter_bounds()
        neutral = default_parameters(1)[0]
        starts = [max(0, case["failure_step"] - offset) for offset in BACKTRACK]
        result = [{"parameters": neutral.tolist(), "mode": 0, "duration": 1,
                   "start": min(starts), "enabled": False}]
        elites = sorted(previous or [], key=lambda row: row["rank"], reverse=True)[:4]
        for k in range(1, count):
            theta = neutral.clone()
            if elites and k <= 2:
                result.append(copy.deepcopy(elites[k - 1]["program"]))
                continue
            if elites:
                bank = torch.tensor([e["program"]["parameters"] for e in elites])
                mean = bank.mean(0)
                std = torch.maximum(bank.std(0, unbiased=False), (high - low) * .08)
                theta = (mean + std * torch.randn(12, generator=generator)).clamp(low, high)
                chosen = elites[int(torch.randint(len(elites), (), generator=generator))]["program"]
                mode = chosen["mode"] if k % 4 else int(torch.randint(5, (), generator=generator))
                start = chosen["start"] if k % 3 else starts[k % len(starts)]
                duration = max(16, min(512, round(chosen["duration"] * float(torch.exp(.3 * torch.randn((), generator=generator))))))
            else:
                # Small feedback perturbations plus discrete engine interventions.
                theta[0] = .12 * torch.randn((), generator=generator)
                theta[1] = .04 * torch.rand((), generator=generator)
                theta[2] = .25 + .35 * torch.rand((), generator=generator)
                theta[3:5] = torch.tensor([.5, 1.]) * torch.rand(2, generator=generator)
                theta[5:7] = torch.tensor([.16, .24]) * torch.rand(2, generator=generator)
                theta[7:9] = .08 * torch.rand(2, generator=generator)
                theta[9:11] = .06 * torch.randn(2, generator=generator)
                theta[11] = 80.  # neutral rate limiter; search can subsequently lower it
                theta = theta.clamp(low, high)
                mode = (k - 1) % 5
                start = starts[(k - 1) % len(starts)]
                duration = (32, 64, 128, 256, 512)[(k // 3) % 5]
                # Isolate engine-mode proposals from attitude perturbations.
                if k <= 5:
                    theta = neutral.clone()
                    if mode == 3:
                        theta[2] = .35
            result.append({"parameters": theta.tolist(), "mode": mode, "duration": duration,
                           "start": start, "enabled": True})
        return result

    def search(self, cases):
        torch = self.torch
        generator = torch.Generator().manual_seed(741007)
        histories = [[] for _ in cases]
        allocation = [16] * len(cases)
        allocation_log = []
        capacity = min(128, self.args.max_envs)
        throughput = None
        for round_index in range(2):
            populations = [self.make_population(case, allocation[i], generator, histories[i]) for i, case in enumerate(cases)]
            allocation_log.append({"round": round_index + 1, "candidates_per_case": allocation.copy(),
                                   "maximum_batch_environments": capacity})
            cursor = batch_index = 0
            while cursor < len(cases):
                group_ids, count = [], 0
                while cursor < len(cases) and (not group_ids or count + 2 * len(populations[cursor]) <= capacity):
                    group_ids.append(cursor); count += 2 * len(populations[cursor]); cursor += 1
                streams, programs, references, splits, mapping = [], [], [], [], []
                for case_id in group_ids:
                    case, population = cases[case_id], populations[case_id]
                    group_start = len(streams)
                    earliest = min(p["start"] for p in population)
                    for candidate_id, program in enumerate(population):
                        for disturbance in range(2):
                            streams.append(case["stream"])
                            programs.append(program)
                            references.append(group_start + disturbance)
                            splits.append((earliest, case["stream"] + (8_000_000 if disturbance else 0)))
                            mapping.append((case_id, candidate_id, disturbance))
                name = f"search-r{round_index + 1}-b{batch_index:02d}"
                result, artifact = self.rollout(name, streams, programs=programs, references=references, splits=splits)
                batch_index += 1
                lookup = {key: row for row, key in enumerate(mapping)}
                for case_id in group_ids:
                    parents = [result["episodes"][lookup[(case_id, 0, d)]] for d in range(2)]
                    for candidate_id, program in enumerate(populations[case_id]):
                        rows = [lookup[(case_id, candidate_id, d)] for d in range(2)]
                        outcomes = [result["episodes"][r] for r in rows]
                        comparison = paired_summary(parents, outcomes)
                        success = comparison["candidate_successes"]
                        quality = sum(r["search_tiebreak"] for r in outcomes) / 2
                        # Lexicographic success first. Tiebreak cannot create success.
                        rank = [success, comparison["net"], quality, -program["duration"]]
                        histories[case_id].append({"round": round_index + 1, "rollout": name,
                            "candidate": candidate_id, "program": copy.deepcopy(program), "rank": rank,
                            "comparison": comparison, "episodes": outcomes,
                            "branch_seen": artifact["branch_seen"][rows].tolist(),
                            "branch_features": artifact["branch_features"][rows].tolist(),
                            "branch_observations": artifact["branch_observations"][rows].tolist()})
                current_throughput = count * HORIZON / max(1e-6, result["wall_seconds"])
                if throughput is None or current_throughput >= .85 * throughput:
                    capacity = min(self.args.max_envs, max(capacity, capacity * 2))
                else:
                    capacity = max(64, capacity // 2)
                throughput = current_throughput
                self.save(search_round=round_index + 1, searched_cases=cursor,
                          next_batch_capacity=capacity, last_search_transitions_per_second=throughput)
                baseline.write_json(self.out / "search_history.json", {"cases": cases, "histories": histories,
                                    "allocations": allocation_log})
            if round_index == 0 and cases:
                weights = []
                for history in histories:
                    best = max(history, key=lambda x: x["rank"])
                    # Promise is empirical paired gain, with a floor for all cases.
                    weights.append(1. + max(0, best["comparison"]["net"]) +
                                   (0.5 if best["comparison"]["candidate_successes"] == 1 else 0.))
                raw = torch.tensor(weights) / sum(weights) * (8 * len(cases))
                extra = raw.floor().long()
                for index in torch.argsort(raw - extra, descending=True)[:8 * len(cases) - int(extra.sum())]:
                    extra[index] += 1
                allocation = [8 + int(x) for x in extra]
        selected = []
        for case_id, history in enumerate(histories):
            options = [r for r in history if r["program"]["enabled"] and
                       r["comparison"]["net"] > 0 and r["comparison"]["candidate_successes"] > 0
                       and r["episodes"][0]["official_success"] and r["branch_seen"][0]]
            if options:
                row = copy.deepcopy(max(options, key=lambda r: r["rank"]))
                row["case_id"] = case_id
                selected.append(row)
        summary = {"selected_programs": len(selected), "failure_cases": len(cases),
                   "robust_two_disturbance_repairs": sum(r["comparison"]["candidate_successes"] == 2 for r in selected),
                   "selected": selected, "allocations": allocation_log,
                   "search_is_training_not_development": True}
        baseline.write_json(self.out / "search_summary.json", summary)
        emit("search_complete", selected_programs=len(selected), failure_cases=len(cases),
             robust_two_disturbance_repairs=summary["robust_two_disturbance_repairs"])
        return selected

    def fit_student(self, screening, traces, selected, extra=None, suffix="initial"):
        torch = self.torch
        from feedback_program import repair_features, FEATURE_DIM
        from repair_policy import RepairGate, CounterfactualRepairPolicy
        device = next(self.parent.parameters()).device
        torch.manual_seed(751007)
        keep = torch.tensor([r["official_success"] for r in screening])
        observations = traces["observations"][::8]
        actions = traces["parent_actions"][::8]
        active = traces["first_episode_active"][::8]
        mask = active & keep[None, :] & (observations[:, :, 12] - 2.575 <= 150.)
        negatives = repair_features(observations[mask].to(device), actions[mask].to(device))
        positives, labels, centers = [], [], []
        for option, row in enumerate(selected):
            valid = [self.torch.tensor(f, device=device) for f, seen, episode in
                     zip(row["branch_features"], row["branch_seen"], row["episodes"])
                     if seen and episode["official_success"]]
            positives.extend(valid)
            labels.extend([option + 1] * len(valid))
            centers.append(torch.stack(valid).mean(0))
        if selected:
            features = torch.cat((negatives, torch.stack(positives)))
            targets = torch.cat((torch.zeros(len(negatives), dtype=torch.long, device=device),
                                 torch.tensor(labels, dtype=torch.long, device=device)))
            if extra:
                extra_features = torch.tensor([x[0] for x in extra], device=device)
                extra_labels = torch.tensor([x[1] for x in extra], dtype=torch.long, device=device)
                # Replication changes training emphasis, not environment accounting.
                features = torch.cat((features, extra_features.repeat(16, 1)))
                targets = torch.cat((targets, extra_labels.repeat(16)))
            gate = RepairGate(len(selected)).to(device)
            tick = time.perf_counter()
            report = gate.fit(features, targets, updates=800)
            report["wall_seconds"] = time.perf_counter() - tick
            parameters = torch.tensor([r["program"]["parameters"] for r in selected], device=device)
            modes = torch.tensor([r["program"]["mode"] for r in selected], device=device)
            durations = torch.tensor([r["program"]["duration"] for r in selected], device=device)
            centers = torch.stack(centers)
        else:
            gate = RepairGate(0).to(device).eval().requires_grad_(False)
            parameters, centers = torch.empty((0, 12), device=device), torch.empty((0, FEATURE_DIM), device=device)
            modes = durations = torch.empty(0, dtype=torch.long, device=device)
            report = {"updates": 0, "threshold": 1., "reason": "no verified new successful program"}
        student = CounterfactualRepairPolicy(self.parent, gate, parameters, modes, durations, centers,
                                            threshold=report["threshold"], support_radius=1.5)
        baseline.write_json(self.out / f"distillation-{suffix}.json", report)
        self.save_checkpoint(student, f"student-{suffix}.pt")
        emit("student_fitted", phase=suffix, programs=len(selected), **report)
        return student

    def save_checkpoint(self, policy, filename):
        self.torch.save({"format": "rocket_counterfactual_repair_v1",
            "parent_payload": self.torch.load(self.out / "parent_checkpoint.pt", weights_only=False, map_location="cpu"),
            "gate_state_dict": policy.gate.state_dict(), "gate_programs": policy.gate.programs,
            "gate_width": policy.gate.width, "parameters": policy.parameters_library.cpu(),
            "modes": policy.modes.cpu(), "durations": policy.durations.cpu(), "centers": policy.centers.cpu(),
            "threshold": policy.threshold, "support_radius": policy.support_radius,
            "training_uses_development_labels": False, "anchor_commit": baseline.ANCHOR}, self.out / filename)

    def paired_student(self, student, seed, count, prefix):
        comparisons, extras, all_parent, all_student = [], [], [], []
        chunk = min(64, self.args.max_envs // 2)
        for start in range(0, count, chunk):
            size = min(chunk, count - start)
            streams = [seed + start + i for i in range(size) for _ in range(2)]
            result, _ = self.rollout(f"{prefix}-{start:03d}", streams, student=student,
                                     student_mask=[i % 2 == 1 for i in range(2 * size)])
            parents, students = result["episodes"][::2], result["episodes"][1::2]
            comparison = paired_summary(parents, students)
            comparisons.append(comparison)
            all_parent.extend(parents); all_student.extend(students)
            for a, b in zip(parents, students):
                trigger = b["intervention"]
                if trigger is not None:
                    # Only actual net gains reinforce an intervention; preserving
                    # existing successes does not invent a positive treatment effect.
                    label = trigger["program"] + 1 if b["official_success"] and not a["official_success"] else 0
                    extras.append((trigger["features"], label))
        summary = paired_summary(all_parent, all_student)
        summary.update(triggered_episodes=len(extras), positive_trigger_outcomes=sum(label > 0 for _, label in extras),
                       parent_settled=sum(r["settled_now"] for r in all_parent),
                       student_settled=sum(r["settled_now"] for r in all_student))
        baseline.write_json(self.out / f"{prefix}-comparison.json", summary)
        emit("paired_student_complete", phase=prefix, **summary)
        return summary, extras

    def evaluate(self, student, seeds=baseline.DEV_SEEDS):
        from metrics import FirstEpisodes
        frozen_hash = baseline.sha256(self.out / "student-final.pt")
        combined, cohorts, reference = FirstEpisodes(128 * len(seeds)), [], []
        for group, seed in enumerate(seeds):
            report, _ = self.rollout(f"development-{seed}", list(range(seed, seed + 128)),
                                     category="evaluation", student=student)
            cohorts.append({k: report[k] for k in ("initial_state_sha256", "summary", "wall_seconds")})
            for j, record in enumerate(report["episodes"]):
                combined.add(group * 128 + j, {k: v for k, v in record.items() if k != "replica"})
            reference_file = self.parent_dir / f"development-{seed}-selector.json"
            old = json.loads(reference_file.read_text())
            if old["initial_state_sha256"] != report["initial_state_sha256"]:
                raise RuntimeError("development parent initial states differ")
            if old["engine_dtype"] != report["engine_dtype"] or self.args.backend != "cpu":
                raise RuntimeError("stored parent comparison requires same CPU backend and dtype")
            reference.extend(old["episodes"])
        comparison = paired_summary(reference, combined.records)
        result = {"summary": combined.summary(), "episodes": combined.records, "cohorts": cohorts,
                  "comparison": comparison, "checkpoint_sha256": frozen_hash,
                  "protocol": "upstream_source_first_episode_full_horizon_v1",
                  "backend": self.args.backend, "anchor_commit": baseline.ANCHOR,
                  "frozen_before_development": True, "sealed_test_used": False,
                  "full_development_protocol": tuple(seeds) == baseline.DEV_SEEDS,
                  "reference": str(self.parent_dir),
                  "reference_checkpoint_sha256": baseline.sha256(self.out / "parent_checkpoint.pt")}
        if baseline.sha256(self.out / "student-final.pt") != frozen_hash:
            raise RuntimeError("student changed during evaluation")
        baseline.write_json(self.out / f"development{128 * len(seeds)}.json", result)
        emit("development_complete", **comparison, success_rate=result["summary"]["success_rate"],
             final_settled=result["summary"]["final_settled_successes"])
        return result

    def run(self):
        screening, traces, cases = self.screen()
        selected = self.search(cases) if cases else []
        student = self.fit_student(screening, traces, selected)
        neighborhood, extras1 = self.paired_student(student, NEIGHBOR_SEED, 32, "training-neighborhood")
        on_policy, extras2 = self.paired_student(student, STUDENT_SEED, 128, "training-on-policy")
        final_student = self.fit_student(screening, traces, selected, extra=extras1 + extras2, suffix="final")
        self.save(status="student_frozen", selected_programs=len(selected),
                  training_complete_seconds=time.perf_counter() - self.started)
        # One full development assessment, including negative results. No model
        # updates or threshold changes are allowed after opening its results.
        development = self.evaluate(final_student)
        baseline.verify_snapshot(self.out)
        if baseline.sha256(self.out / "parent_checkpoint.pt") != self.manifest["parent_sha256"]:
            raise RuntimeError("parent checkpoint changed")
        report = {"status": "completed", "search_programs": len(selected),
                  "training_neighborhood": neighborhood, "training_on_policy": on_policy,
                  "development": {k: v for k, v in development.items() if k != "episodes"},
                  "training_transitions": self.status["training_transitions"],
                  "evaluation_transitions": self.status["evaluation_transitions"],
                  "environment_transitions": self.status["environment_transitions"],
                  "wall_seconds": time.perf_counter() - self.started,
                  "matched_budget_ppo_control_completed": False,
                  "teacher_calls": 0, "sealed_test_used": False}
        baseline.write_json(self.out / "report.json", report)
        self.save(status="completed", comparison=development["comparison"],
                  success_rate=development["summary"]["success_rate"])
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=baseline.CHECKPOINT)
    parser.add_argument("--backend", choices=("cpu",), default="cpu")
    parser.add_argument("--max-envs", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--physics-threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_envs < 128:
        parser.error("--max-envs must be at least 128 for the unchanged development protocol")
    experiment = None
    try:
        experiment = Experiment(args)
        experiment.run()
    except BaseException as exc:
        if experiment is not None:
            experiment.save(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
