"""Evaluate a saved unified actor in four independent original 128-env cohorts.

Only scheduling changes: immutable simulator/model sources, the same four RNG
seeds, deterministic actions, original termination and first-episode accounting.
Workers load no optimizer or training rollout tensors. Each writes recoverable
episode records and a heartbeat every 16 vector steps.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import snapshot as baseline


def run_cohort(args):
    run, out = args.run_dir.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"status": "initializing", "seed": args.seed, "pid": os.getpid(),
              "num_envs": 128, "started_at_unix": time.time()}

    def emit(event, **fields):
        status.update(fields, event=event, wall_seconds=time.perf_counter()-started)
        baseline.write_json(out/"status.json", status)
        print(json.dumps(status, allow_nan=False), flush=True)

    env = None
    try:
        for key in ("QD_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "1"
        # Independent cache directories prevent competing cache writes at startup.
        os.environ["GS_CACHE_FILE_PATH"] = str(out/"cache/genesis")
        os.environ["MPLCONFIGDIR"] = str(out/"cache/matplotlib")
        torch, _ = baseline.setup_imports(run/"source", out, 1)
        from common.rocket_phase_policy import RocketPhaseModel
        from common.rocket_unified_physics import UnifiedRocketActor, PhysicalMemory
        from common.dynamics import euler
        from pyflyt_genesis.rocket import RocketLandingEnv
        from episode_metrics import FirstEpisodes
        import genesis as gs

        checkpoint = torch.load(run/"checkpoint.pt", map_location="cpu", weights_only=False)
        configuration = checkpoint["manifest"]["configuration"]
        torch.manual_seed(configuration["seed"])
        env = RocketLandingEnv(num_envs=128, backend=args.backend, seed=args.seed, difficulty=None)
        if args.backend == "gpu" and gs.tc_float != torch.float32:
            raise RuntimeError("original evaluation engine must remain float32")
        observation = env.get_observations()["policy"]
        cfg = json.loads((run/"parent_config.json").read_text())
        params = copy.deepcopy(cfg["actor"])
        params.pop("class_name")
        std = params.get("distribution_cfg", {}).get("init_std")
        if isinstance(std, list):
            params["distribution_cfg"]["init_std"] = torch.tensor(std, device=env.device)
        seed_policy = RocketPhaseModel({"policy": observation}, cfg["obs_groups"], "actor", 7,
                                      **params).to(env.device)
        actor = UnifiedRocketActor(seed_policy, horizon=configuration["horizon"],
                                   iterations=configuration["iterations"]).to(env.device)
        actor.load_state_dict(checkpoint["actor"], strict=True)
        actor.eval()
        del checkpoint
        memory = PhysicalMemory(actor, 128, env.device)
        initial = torch.cat(env.plant.state(), -1).detach().cpu()
        initial_hash = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
        book = FirstEpisodes(128)
        pending = torch.ones(128, dtype=torch.bool, device=env.device)
        completed = 0
        original_reset = env.reset_idx

        def record_then_reset(mask):
            nonlocal completed
            first = mask & pending
            ids = first.nonzero().flatten().tolist()
            if ids:
                _, q, v, w = env.plant.state()
                tilt = euler(q)[:, :2].norm(dim=-1)
                settled = (env.plant.pad_contact & ~env.fatal_collision &
                           (v.norm(dim=-1)<.02) & (w.norm(dim=-1)<.02) & (tilt<.1))
                fields = {"official_success": env.complete, "fatal_collision": env.fatal_collision,
                          "out_of_bounds": env.out_of_bounds, "settled_now": settled,
                          "episode_length": env.count, "terminal_speed": v.norm(dim=-1)}
                values = {k:v[first].detach().cpu().tolist() for k,v in fields.items()}
                for j, i in enumerate(ids):
                    row = {k:v[j] for k,v in values.items()}
                    row.update(sample_id=f"iid-stream-{args.seed+i}", rng_stream_id=args.seed+i,
                               initial_state_p_q_v_w=initial[i].tolist(),
                               success_then_failure=bool(row["official_success"] and
                                   (row["fatal_collision"] or row["out_of_bounds"])),
                               failure_reason=None if row["official_success"] else
                                   "fatal_collision" if row["fatal_collision"] else
                                   "out_of_bounds" if row["out_of_bounds"] else "timeout")
                    completed += int(book.add(i, row))
                pending[first] = False
            original_reset(mask)

        env.reset_idx = record_then_reset

        def choose_action():
            # Autograd is required INSIDE the physical solver even in evaluation.
            # Its temporary graphs die here, before advancing the real simulator.
            result = actor(observation, **memory.inputs(), create_graph=False)
            with torch.no_grad():
                return actor.action(result, deterministic=True).detach(), result["belief"]["memory"].detach()

        def sync():
            if args.backend == "gpu":
                torch.cuda.synchronize()

        sync()
        build_seconds = time.perf_counter()-started
        loop_started = time.perf_counter()
        policy_seconds = simulation_seconds = 0.
        profiled_steps = 0
        emit("evaluation_started", status="evaluating", vector_steps=0,
             max_vector_steps=int(env.max_steps), episodes_finished=0,
             initial_state_sha256=initial_hash, build_seconds=build_seconds)
        for t in range(int(env.max_steps)):
            profile = t < 4 or t % 64 == 0
            if profile:
                sync()
                begin = time.perf_counter()
            action, next_memory = choose_action()
            if profile:
                sync()
                middle = time.perf_counter()
            with torch.no_grad():
                following, _, done, _ = env.step(action)
                memory.advance(observation, action, next_memory, done)
                observation = following["policy"]
            if profile:
                sync()
                policy_seconds += middle-begin
                simulation_seconds += time.perf_counter()-middle
                profiled_steps += 1
            if (t+1) % 16 == 0 or completed == 128 or t+1 == int(env.max_steps):
                sync()
                elapsed = time.perf_counter()-loop_started
                details = {"vector_steps": t+1, "max_vector_steps": int(env.max_steps),
                           "episodes_finished": completed, "loop_seconds": elapsed,
                           "vector_steps_per_second": (t+1)/max(elapsed, 1e-9),
                           "profiled_steps": profiled_steps,
                           "policy_seconds_per_profiled_step": policy_seconds/max(1,profiled_steps),
                           "simulation_seconds_per_profiled_step": simulation_seconds/max(1,profiled_steps)}
                if args.backend == "gpu":
                    details.update(gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                                   gpu_reserved_gib=torch.cuda.memory_reserved()/2**30)
                emit("evaluation_progress", **details)
                baseline.write_json(out/"partial_episodes.json", {
                    "seed": args.seed, "vector_steps": t+1, "summary": book.summary(),
                    "episodes": [row for row in book.records if row is not None]})
            # No early success shortcut: stop only after ALL first episodes ended
            # under the unchanged environment's own termination rules.
            if completed == 128:
                break
        report = {"seed": args.seed, "summary": book.summary(), "episodes": book.records,
                  "initial_state_sha256": initial_hash,
                  "checkpoint_sha256": baseline.sha256(run/"checkpoint.pt"),
                  "wall_seconds": time.perf_counter()-started,
                  "timing": {"profiled_steps": profiled_steps,
                             "policy_seconds": policy_seconds, "simulation_seconds": simulation_seconds}}
        baseline.write_json(out/"report.json", report)
        if not report["summary"]["complete_evaluation"]:
            raise RuntimeError("original full horizon left missing first episodes")
        emit("evaluation_cohort", status="completed", summary=report["summary"])
    except BaseException as error:
        emit("error", status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()


def coordinate(args):
    run, out = args.run_dir.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline.verify_snapshot(run)
    manifest = json.loads((run/"manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        if baseline.sha256(run/"source/src/common"/name) != digest:
            raise RuntimeError(f"saved actor source changed: {name}")
    checkpoint_hash = baseline.sha256(run/"checkpoint.pt")
    shutil.copy2(__file__, out/"evaluator.py")
    baseline.write_json(out/"manifest.json", {
        "checkpoint_sha256": checkpoint_hash, "run_dir": str(run),
        "source": str(run/"source"), "seeds": list(baseline.DEV_SEEDS),
        "num_envs_per_cohort": 128, "workers": args.workers,
        "engine_precision": "float32", "full_original_horizon": True,
        "physics_and_success_predicate_modified": False,
        "policy_or_solver_modified": False, "sealed_test_used": False,
        "evaluator_sha256": baseline.sha256(Path(__file__))})
    baseline.write_json(run/"evaluation_pointer.json", {
        "out": str(out), "checkpoint_sha256": checkpoint_hash, "workers": args.workers})
    pending = list(baseline.DEV_SEEDS)
    active = {}
    finished = []
    started = time.perf_counter()
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                seed = pending.pop(0)
                child_out = out/f"seed_{seed}"
                child_out.mkdir()
                log = (child_out/"worker.log").open("w")
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--run-dir", str(run), "--out", str(child_out),
                           "--backend", args.backend, "--seed", str(seed)]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                log.close()
                active[seed] = process
            for seed, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f"evaluation seed {seed} exited {code}; see its worker.log")
                finished.append(seed)
                del active[seed]
            children = {}
            for seed in baseline.DEV_SEEDS:
                path = out/f"seed_{seed}"/"status.json"
                if path.exists():
                    children[str(seed)] = json.loads(path.read_text())
            baseline.write_json(out/"status.json", {
                "status": "evaluating", "wall_seconds": time.perf_counter()-started,
                "completed_cohorts": len(finished), "planned_cohorts": 4,
                "children": children})
            if pending or active:
                time.sleep(2)
        from episode_metrics import FirstEpisodes
        book = FirstEpisodes(512)
        cohorts, records = [], []
        for seed in baseline.DEV_SEEDS:
            report = json.loads((out/f"seed_{seed}"/"report.json").read_text())
            if report["checkpoint_sha256"] != checkpoint_hash or not report["summary"]["complete_evaluation"]:
                raise RuntimeError(f"invalid completed cohort {seed}")
            cohorts.append({k:report[k] for k in ("seed", "summary", "initial_state_sha256")})
            records.extend(report["episodes"])
        if len(records) != 512 or len({r["sample_id"] for r in records}) != 512:
            raise RuntimeError("expected 512 distinct fixed first episodes")
        for i, row in enumerate(records):
            book.add(i, row)
        report = {"summary": book.summary(), "cohorts": cohorts, "episodes": records,
                  "checkpoint_sha256": checkpoint_hash, "sealed_test_used": False,
                  "physics_and_success_predicate_modified": False,
                  "evaluation_schedule": "four independent 128-env processes",
                  "wall_seconds": time.perf_counter()-started}
        if baseline.sha256(run/"checkpoint.pt") != checkpoint_hash:
            raise RuntimeError("checkpoint changed during evaluation")
        baseline.write_json(out/"development512.json", report)
        baseline.write_json(run/"development512.json", report)
        final = {"status": "completed", "wall_seconds": report["wall_seconds"],
                 "completed_cohorts": 4, "planned_cohorts": 4, "evaluation_summary": report["summary"]}
        baseline.write_json(out/"status.json", final)
        training_status = json.loads((run/"status.json").read_text())
        training_status.update(status="completed", last_event="evaluation_completed",
                               evaluation_directory=str(out), evaluation_summary=report["summary"])
        baseline.write_json(run/"status.json", training_status)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        for process in active.values():
            if process.poll() is None:
                process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        baseline.write_json(out/"status.json", {
            "status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--workers", type=int, choices=range(1,5), default=4)
    parser.add_argument("--seed", type=int, choices=baseline.DEV_SEEDS)
    args = parser.parse_args()
    run_cohort(args) if args.seed is not None else coordinate(args)


if __name__ == "__main__":
    main()
