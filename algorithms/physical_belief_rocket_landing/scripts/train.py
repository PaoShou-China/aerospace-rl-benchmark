"""Train the complete recurrent physical actor with real-environment PPO.

One campaign, no intermediate success-rate screening: snapshot, warm start,
causal physical belief + contact + robust hybrid control + PPO + sequence
prediction, checkpoint, then one unchanged 512-episode evaluation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import snapshot as baseline


OVERLAYS = ("rocket_phase_policy.py", "rocket_physical_core.py", "rocket_unified_physics.py",
            "rocket_flash_ppo.py", "rocket_reward_candidates.py", "rocket_direct_support.py",
            "rocket_ppo_practical_support.py", "rocket_confident_physics.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=baseline.ROOT/"artifacts/parent")
    parser.add_argument("--checkpoint", default="parent.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--architecture", choices=("published", "confidence_sequence"), default="published")
    parser.add_argument("--hypotheses", type=int, default=7)
    parser.add_argument("--knots", type=int, default=3)
    parser.add_argument("--ablation", choices=("full", "fixed_belief", "no_solver"), default="full")
    parser.add_argument("--disable-confidence", action="store_true")
    parser.add_argument("--reward-variant", choices=("settle", "persistent"), default="settle")
    parser.add_argument("--prediction-coefficient", type=float, default=.1)
    parser.add_argument("--anchor-coefficient", type=float, default=.002)
    parser.add_argument("--initial-physical-authority", type=float, default=.25)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--total-transitions", type=int,
                        help="Total lineage interaction budget, including the resumed checkpoint")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--inference-batch", type=int, default=512)
    parser.add_argument("--max-microbatch", type=int, default=4096)
    parser.add_argument("--batch-tune-every", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.8)
    parser.add_argument("--gpu-memory-budget-gib", type=float, default=0.)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--target-kl", type=float, default=.02)
    parser.add_argument("--kl-coefficient", type=float, default=1.)
    parser.add_argument("--seed-learning-rate", type=float, default=5e-6)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=700007)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--evaluation-workers", type=int, choices=range(1,5), default=4)
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        parser.error("choose resume or initialize-from")
    if not .5 <= args.gpu_memory_fraction <= .97:
        parser.error("gpu-memory-fraction must be between .5 and .97")
    if min(args.num_envs, args.rollout, args.minibatch, args.microbatch,
           args.inference_batch, args.max_microbatch, args.batch_tune_every) <= 0:
        parser.error("environment and batch sizes must be positive")
    transition_target = args.total_transitions or args.updates*args.rollout*args.num_envs
    if transition_target % args.num_envs:
        parser.error("total transition budget must be divisible by num-envs")
    out, parent = args.out.resolve(), args.parent.resolve()
    source = baseline.prepare_snapshot(out)
    for name in OVERLAYS:
        shutil.copy2(baseline.ROOT/"core"/name, source/"src/common"/name)
    shutil.copy2(__file__, out/"campaign.py")
    shutil.copy2(parent/args.checkpoint, out/"parent.pt")
    cfg = json.loads((parent/"config.json").read_text())
    baseline.write_json(out/"parent_config.json", cfg)
    for key in ("QD_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    torch, _ = baseline.setup_imports(source, out, 1)
    from common.rocket_phase_policy import RocketPhaseModel
    from common.rocket_unified_physics import UnifiedRocketActor, PhysicalMemory
    from common.rocket_confident_physics import build_physical_actor, initialize_from_published
    from common.rocket_ppo_practical_support import PracticalRewardAdapter
    from common.rocket_physical_core import observation_state
    from pyflyt_genesis.rocket import RocketLandingEnv
    import genesis as gs
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    env = None
    status = {"status": "initializing", "started_at_unix": time.time(), "completed_updates": 0,
              "new_environment_transitions": 0, "planned_updates": args.updates,
              "planned_environment_transitions": transition_target,
              "training_success_rate_is_evaluation": False}

    def emit(event, **details):
        row = {"event": event, "wall_seconds": time.perf_counter()-started, **details}
        print(json.dumps(row, allow_nan=False), flush=True)
        with (out/"progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        status.update(details, last_event=event, wall_seconds=row["wall_seconds"])
        baseline.write_json(out/"status.json", status)

    class DynamicBatch:
        def __init__(self):
            self.inference = min(args.num_envs, args.inference_batch)
            self.training = min(args.minibatch, args.max_microbatch, args.microbatch)
            self.minibatch, self.maximum = args.minibatch, args.max_microbatch
            self.memory_fraction = args.gpu_memory_fraction
            self.oom_ceiling = args.max_microbatch
            self.oom_retries = 0
            self.control_path = out/"batch_control.json"
            self.last_control = None
            baseline.write_json(self.control_path, {
                "minibatch":self.minibatch, "microbatch":self.training,
                "max_microbatch":self.maximum, "inference_batch":self.inference,
                "gpu_memory_fraction":self.memory_fraction})

        def refresh(self):
            raw = self.control_path.read_text()
            if raw == self.last_control:
                return
            config = json.loads(raw)
            maximum = args.num_envs*args.rollout
            for key in ("minibatch","microbatch","max_microbatch","inference_batch"):
                if not isinstance(config[key], int) or config[key] < 1 or config[key] > maximum:
                    raise ValueError(f"invalid runtime batch setting: {key}")
            if not .5 <= config["gpu_memory_fraction"] <= .97:
                raise ValueError("invalid runtime GPU memory fraction")
            self.minibatch = config["minibatch"]
            self.maximum = config["max_microbatch"]
            self.training = min(config["microbatch"],self.minibatch,self.maximum,self.oom_ceiling)
            self.inference = min(args.num_envs,config["inference_batch"])
            self.memory_fraction = config["gpu_memory_fraction"]
            self.last_control = raw
            emit("batch_control", batch_configuration=config)

        def shrink(self):
            if self.training <= 256:
                return False
            self.training = max(256, int(self.training*.75)//256*256)
            self.oom_ceiling = min(self.oom_ceiling,self.training)
            self.oom_retries += 1
            return True

        def tune(self):
            if args.backend != "gpu":
                return
            free, total = torch.cuda.mem_get_info()
            peak = torch.cuda.max_memory_allocated()
            persistent = torch.cuda.memory_allocated()
            other_usage = max(0, total-free-torch.cuda.memory_reserved())
            budget = max(0, self.memory_fraction*total-other_usage)
            if args.gpu_memory_budget_gib > 0:
                budget = min(budget, args.gpu_memory_budget_gib*2**30)
            # Estimate activation bytes per sample from real training, keeping
            # persistent tensors and other CUDA processes outside this budget.
            workspace = max(peak-persistent, 256*2**20)
            multiplier = max(.5,min(2.,.97*max(0,budget-persistent)/workspace))
            proposed = max(256,int(self.training*multiplier)//256*256)
            self.training = min(self.minibatch,self.maximum,self.oom_ceiling,proposed)
            torch.cuda.reset_peak_memory_stats()

        def resources(self):
            if args.backend != "gpu":
                return {}
            free, total = torch.cuda.mem_get_info()
            return {"gpu_peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
                    "gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                    "gpu_free_gib":free/2**30, "gpu_total_gib":total/2**30}

    batches = DynamicBatch()

    def detached(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        return {key: detached(value) for key, value in obj.items()}

    def joined(items):
        if isinstance(items[0], torch.Tensor):
            return torch.cat(items, 0)
        return {key: joined([item[key] for item in items]) for key in items[0]}

    def infer(actor, obs, memory):
        pieces = []
        pos = 0
        while pos < len(obs):
            end = min(len(obs), pos+batches.inference)
            try:
                result = actor(obs[pos:end], **{k:v[pos:end] for k,v in memory.items()}, create_graph=False)
                pieces.append(detached(result))
                pos = end
            except torch.cuda.OutOfMemoryError:
                if batches.inference <= 8:
                    raise
                batches.inference = max(8, batches.inference//2)
                batches.oom_retries += 1
                gc.collect()
                torch.cuda.empty_cache()
        return joined(pieces)

    try:
        env = RocketLandingEnv(num_envs=args.num_envs, backend=args.backend, seed=args.seed, difficulty=None)
        device = env.device
        if gs.tc_float != torch.float32:
            raise RuntimeError("the original evaluation/training engine must remain float32")
        obs = env.get_observations()["policy"]
        parent_data = torch.load(out/"parent.pt", weights_only=False, map_location=device)

        def build(role, output_dim):
            params = copy.deepcopy(cfg[role])
            params.pop("class_name")
            std = params.get("distribution_cfg", {}).get("init_std")
            if isinstance(std, list):
                params["distribution_cfg"]["init_std"] = torch.tensor(std, device=device)
            model = RocketPhaseModel({"policy":obs}, cfg["obs_groups"], role, output_dim, **params).to(device)
            model.load_state_dict(parent_data[role+"_state_dict"], strict=True)
            return model

        seed_policy, critic = build("actor", 7), build("critic", 1)
        reference = copy.deepcopy(seed_policy).eval().requires_grad_(False)
        actor = build_physical_actor(seed_policy, vars(args)).to(device)
        seed_ids = {id(p) for p in seed_policy.parameters()}
        new_parameters = [p for p in actor.parameters() if id(p) not in seed_ids]
        optimizer = torch.optim.Adam([
            {"params":seed_policy.parameters(), "lr":args.seed_learning_rate},
            {"params":new_parameters, "lr":args.actor_learning_rate},
            {"params":critic.parameters(), "lr":1e-4}], eps=1e-5)
        first_update = 0
        transferred_keys = []
        if args.initialize_from:
            initial = torch.load(args.initialize_from, map_location=device, weights_only=False)
            transferred_keys = initialize_from_published(actor, initial['actor'])
            critic.load_state_dict(initial['critic'], strict=True)
            status['new_environment_transitions'] = int(initial['environment_transitions'])
            if args.architecture == 'confidence_sequence':
                with torch.no_grad():
                    authority = min(.95, max(.05, args.initial_physical_authority))
                    actor.physical_mix_logit.fill_(math.log(authority/(1-authority)))
        if args.resume:
            resumed = torch.load(args.resume, map_location=device, weights_only=False)
            actor.load_state_dict(resumed["actor"])
            critic.load_state_dict(resumed["critic"])
            optimizer.load_state_dict(resumed["optimizer"])
            optimizer.param_groups[0]["lr"] = args.seed_learning_rate
            optimizer.param_groups[1]["lr"] = args.actor_learning_rate
            first_update = int(resumed["update"])
            status["completed_updates"] = first_update
            old_config = resumed["manifest"]["configuration"]
            status["new_environment_transitions"] = int(resumed.get("environment_transitions",
                first_update*int(old_config["rollout"])*int(old_config["num_envs"])))
        remaining_transitions = transition_target-status["new_environment_transitions"]
        if remaining_transitions < 0 or remaining_transitions % args.num_envs:
            raise ValueError("remaining transition budget must be nonnegative and divisible by num-envs")
        last_update = first_update+math.ceil(remaining_transitions/(args.rollout*args.num_envs))
        status["planned_updates"] = last_update
        manifest = {"architecture": args.architecture,
            "anchor_commit":baseline.ANCHOR, "parent_checkpoint_sha256":baseline.sha256(out/"parent.pt"),
            "source_sha256":{name:baseline.sha256(source/"src/common"/name) for name in OVERLAYS},
            "launcher_sha256":baseline.sha256(out/"campaign.py"),
            "configuration":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "actor_parameters":sum(p.numel() for p in actor.parameters()),
            "new_parameters":sum(p.numel() for p in new_parameters),
            "physical_parameters":actor.physics.parameter_names,
            "components":["causal recurrent memory", "weighted three-hypothesis physical belief",
                          "six-DOF flight", "three-leg compliant contact", "smooth sliding friction",
                          "shared learned ID/control update", "two ignition branches",
                          "differentiable robust action optimization", "final hybrid likelihood PPO",
                          "actual-executed-action prediction loss", "four-step belief TBPTT",
                          "adaptive CUDA microbatches", "checkpoint/resume", "original final evaluation"],
            "reward":args.reward_variant,
            "training_success_early_termination":args.reward_variant == 'settle', "physical_environment_modified":False,
            "evaluation_success_predicate_modified":False, "teacher_queries":0,
            "hypothesis_uncertainty_calibrated":False, "contact_is_differentiable_surrogate":True,
            "internal_physics":"40Hz macro steps; analytic six-tick actuator endpoint/mean and regularized contact",
            "actual_simulator_physics_hz":240,
            "resume_semantics":"weights and optimizer restored; fresh simulator episodes, not bit-exact physics resume",
            "resume_completed_updates":first_update,
            "resume_environment_transitions":status["new_environment_transitions"],
            "planned_total_updates":last_update,
            "planned_total_environment_transitions":transition_target,
            "new_process_interaction_budget":remaining_transitions,
            "runtime_batch_control":str(batches.control_path),
            "gpu_memory_fraction":args.gpu_memory_fraction,
            "microbatch_graph_lifetime":"helper scope; release rejected forwards before continuing",
            "training_episode_counters_scope":"current process only",
            "sealed_test_used":False, "intermediate_evaluations":0,
            "ppo_trust_region":"exact joint Bernoulli/conditional Normal KL penalty and minibatch early stop",
            "resume_checkpoint_sha256":baseline.sha256(args.resume) if args.resume else None,
            "initialization_checkpoint_sha256":baseline.sha256(args.initialize_from) if args.initialize_from else None,
            "transferred_parameter_keys":transferred_keys,
            "initialization_optimizer_reset":bool(args.initialize_from),
            "reliability_score_is_probability":False}
        baseline.write_json(out/"manifest.json", manifest)
        memory = PhysicalMemory(actor, args.num_envs, device)
        # A hold window longer than the original episode disables early
        # success resets while retaining the recorded reward implementation.
        hold_steps = int(env.max_steps)+1 if args.reward_variant == 'persistent' else 64
        adapter = PracticalRewardAdapter(env, gamma=.999, reward_variant='settle', success_hold_steps=hold_steps)
        emit("training_started", status="training", actor_parameters=manifest["actor_parameters"],
             new_parameters=manifest["new_parameters"], num_envs=args.num_envs,
             horizon=args.horizon, shared_iterations=args.iterations,
             resumed_environment_transitions=status["new_environment_transitions"],
             this_process_interaction_budget=remaining_transitions)

        def checkpoint(update, name):
            target = out/name
            torch.save({"actor":actor.state_dict(), "critic":critic.state_dict(),
                        "optimizer":optimizer.state_dict(), "update":update, "manifest":manifest,
                        "environment_transitions":status["new_environment_transitions"],
                        "torch_rng":torch.get_rng_state(),
                        "cuda_rng":torch.cuda.get_rng_state_all() if args.backend == "gpu" else None},
                       target.with_suffix(".tmp"))
            target.with_suffix(".tmp").replace(target)

        def train_micro(b, fraction, previous_kl, previous_rows):
            # Keep the full higher-order physics graph inside this scope. In
            # particular, a KL-rejected forward must not survive until the next
            # forward or the recurrent auxiliary update.
            result = actor(b["obs"], **{k:b[k] for k in
                ("memory","previous_state","executed_action","history_valid")})
            lp = actor.log_prob(result, b["action"])
            exact_kl = actor.kl_divergence(
                {k:b["old_"+k] for k in ("logits","means","scales")}, result).mean()
            count = len(b["obs"])
            kl_sum = previous_kl+float(exact_kl.detach())*count
            rows_seen = previous_rows+count
            if kl_sum/rows_seen > 1.5*args.target_kl:
                return None, kl_sum, rows_seen
            ratio = (lp-b["log_prob"]).clamp(-15,15).exp()
            policy_loss = -torch.minimum(ratio*b["advantage"],
                ratio.clamp(.9,1.1)*b["advantage"]).mean()
            value = critic({"policy":b["obs"]}).squeeze(-1)
            value_loss = nn_huber(torch, value, b["returns"])
            prediction = actor.prediction_loss(result["belief"],b["action"],b["next_obs"],~b["done"])
            entropy = actor.entropy(result).mean()
            with torch.no_grad():
                prior = reference({"policy":b["obs"]})[:,actor.continuous]
            anchoring = ((result["means"]-prior[:,None])/result["scales"]).square().mean()
            objective = (policy_loss+.5*value_loss+args.prediction_coefficient*prediction-.001*entropy
                         +args.anchor_coefficient*anchoring+.0001*result["model_control_cost"].mean()
                         +args.kl_coefficient*exact_kl)
            (objective*fraction).backward()
            with torch.no_grad():
                approximate_kl = ((ratio-1)-(lp-b["log_prob"])).mean()
                metrics = torch.stack((policy_loss,value_loss,prediction,entropy,anchoring,
                                       approximate_kl,exact_kl)).detach()*count
            return metrics, kl_sum, rows_seen

        gamma, lam = .999, .98
        for update in range(first_update, last_update):
            begin = time.perf_counter()
            batches.refresh()
            vector_steps = min(args.rollout, (transition_target-status["new_environment_transitions"])//args.num_envs)
            used_inference_batch = batches.inference
            rows = []
            physical_diagnostics = []
            for t in range(vector_steps):
                memory_input = memory.inputs()
                result = infer(actor, obs, memory_input)
                with torch.no_grad():
                    action = actor.action(result)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("nonfinite policy action")
                    value = critic({"policy":obs}).squeeze(-1)
                    log_prob = actor.log_prob(result, action)
                    next_td, reward, done, extras = adapter.step(action)
                    done = done.clone()
                    next_obs = next_td["policy"]
                    terminal = torch.where(done[:, None], extras["terminal_observations"]["policy"], next_obs)
                    if args.architecture == 'confidence_sequence' and t % 16 == 0:
                        prediction, variance = actor.prediction_statistics(result['belief'], action)
                        error = actor.physics.residual(prediction, observation_state(terminal))
                        physical_diagnostics.append(torch.stack((result['confidence'].mean(),
                            result['solver_improvement'].mean(), result['control_acceptance'].mean(),
                            result['physical_action_delta'].mean(), error.square().mean(),
                            (error.abs() <= 1.96*variance.sqrt()).float().mean())))
                    terminal_value = critic({"policy":terminal}).squeeze(-1)
                    # Bootstrap true timeout from its pre-reset state, never from reset.
                    adjusted = reward+gamma*terminal_value*extras["time_outs"].float()
                    rows.append({"obs":obs.clone(), "action":action, "log_prob":log_prob,
                                 "value":value, "reward":adjusted, "done":done,
                                 "next_obs":terminal.clone(),
                                 "old_logits":result["logits"].clone(),
                                 "old_means":result["means"].clone(),
                                 "old_scales":result["scales"].clone(), **memory_input})
                    memory.advance(obs, action, result["belief"]["memory"], done)
                    obs = next_obs
                status["new_environment_transitions"] += args.num_envs
                if (t+1) % 16 == 0:
                    emit("collecting", current_update=update+1, collected_vector_steps=t+1,
                         new_environment_transitions=status["new_environment_transitions"])
            collection_seconds = time.perf_counter()-begin
            data = {k:torch.stack([row[k] for row in rows]) for k in rows[0]}
            with torch.no_grad():
                next_value = critic({"policy":obs}).squeeze(-1)
                advantages = torch.zeros_like(data["reward"])
                gae = torch.zeros_like(next_value)
                for t in reversed(range(vector_steps)):
                    live = (~data["done"][t]).float()
                    delta = data["reward"][t]+gamma*next_value*live-data["value"][t]
                    gae = delta+gamma*lam*live*gae
                    advantages[t] = gae
                    next_value = data["value"][t]
                returns = advantages+data["value"]
                advantages = (advantages-advantages.mean())/advantages.std().clamp_min(1e-6)
            flat = {k:v.flatten(0,1) for k,v in data.items()}
            flat["advantage"], flat["returns"] = advantages.flatten(), returns.flatten()
            total = len(flat["obs"])
            metric_sums = torch.zeros(7, device=device)
            loss_count = 0
            optimizer_steps = 0
            stopped_for_kl = False
            stop_kl = 0.
            optimization_start = time.perf_counter()
            for epoch in range(args.epochs):
                order = torch.randperm(total, device=device)
                for indices in order.split(batches.minibatch):
                    while True:
                        optimizer.zero_grad(set_to_none=True)
                        current_metrics = torch.zeros_like(metric_sums)
                        batch_kl = 0.
                        batch_rows = 0
                        retry_oom = False
                        try:
                            for micro in indices.split(batches.training):
                                b = {k:v[micro] for k,v in flat.items()}
                                micro_metrics,batch_kl,batch_rows = train_micro(
                                    b,len(micro)/len(indices),batch_kl,batch_rows)
                                if micro_metrics is None:
                                    stopped_for_kl = True
                                    stop_kl = batch_kl/batch_rows
                                    optimizer.zero_grad(set_to_none=True)
                                    break
                                current_metrics += micro_metrics
                            if stopped_for_kl:
                                break
                            norm = torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.)
                            if not torch.isfinite(norm):
                                raise RuntimeError("nonfinite end-to-end PPO gradient")
                            optimizer.step()
                            optimizer_steps += 1
                            metric_sums += current_metrics
                            loss_count += len(indices)
                            break
                        except torch.cuda.OutOfMemoryError:
                            if not batches.shrink():
                                raise
                            retry_oom = True
                        if retry_oom:
                            # Release the failed helper's traceback before cache
                            # reclamation, then retry the uncommitted minibatch.
                            optimizer.zero_grad(set_to_none=True)
                            b = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            emit("microbatch_reduced",microbatch=batches.training,
                                 oom_retries=batches.oom_retries)
                    if stopped_for_kl:
                        break
                if stopped_for_kl:
                    break
            # One TBPTT update on a real four-transition sequence. Belief and
            # recurrent hidden stay tensors across time; resets explicitly cut them.
            sequence_env = torch.randperm(args.num_envs, device=device)[:min(16,args.num_envs)]
            sequence_start = int(torch.randint(max(1,vector_steps-3),(1,)))
            sequence_memory = data["memory"][sequence_start,sequence_env].detach()
            sequence_loss = torch.zeros((),device=device)
            steps = min(4,vector_steps-sequence_start)
            for t in range(sequence_start,sequence_start+steps):
                belief = actor.infer_belief(data["obs"][t,sequence_env], sequence_memory,
                    data["previous_state"][t,sequence_env],data["executed_action"][t,sequence_env],
                    data["history_valid"][t,sequence_env])
                sequence_loss = sequence_loss+actor.prediction_loss(belief, data["action"][t,sequence_env],
                    data["next_obs"][t,sequence_env],~data["done"][t,sequence_env])/steps
                sequence_memory = torch.where(data["done"][t,sequence_env,None],
                    actor.initial_memory(len(sequence_env),device),belief["memory"])
            optimizer.zero_grad(set_to_none=True)
            (args.prediction_coefficient*sequence_loss).backward()
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite recurrent belief gradient")
            optimizer.step()
            optimization_seconds = time.perf_counter()-optimization_start
            metrics = (metric_sums/max(1,loss_count)).tolist()
            used_training_batch = batches.training
            resources = batches.resources()
            if physical_diagnostics:
                values = torch.stack(physical_diagnostics).mean(0).tolist()
                resources.update(dict(zip(('physical_confidence', 'solver_relative_improvement',
                    'control_update_acceptance', 'physical_action_delta', 'normalized_prediction_mse',
                    'prediction_interval_coverage_95'), values)))
            if (update+1-first_update)%args.batch_tune_every == 0:
                batches.tune()
            emit("update", completed_updates=update+1, current_update=update+1,
                 collection_seconds=collection_seconds, optimization_seconds=optimization_seconds,
                 transitions_per_wall_second=args.num_envs*vector_steps/(time.perf_counter()-begin),
                 collected_transitions=args.num_envs*vector_steps,
                 optimization_samples=loss_count,
                 optimization_samples_per_second=loss_count/max(optimization_seconds,1e-9),
                 policy_loss=metrics[0],value_loss=metrics[1],prediction_loss=metrics[2],
                 entropy=metrics[3],anchor_cost=metrics[4],approximate_kl=metrics[5],
                 exact_policy_kl=metrics[6],optimizer_steps=optimizer_steps,
                 kl_early_stop=stopped_for_kl,stop_kl=stop_kl,
                 sequence_prediction_loss=float(sequence_loss.detach()),
                 completed_training_episodes=int(adapter.completed_episodes.sum()),
                 completed_training_successes=int(adapter.completed_successes.sum()),
                 inference_batch=used_inference_batch,microbatch=used_training_batch,
                 minibatch=batches.minibatch,gpu_memory_fraction=batches.memory_fraction,
                 next_inference_batch=batches.inference,next_microbatch=batches.training,
                 oom_retries=batches.oom_retries,**resources)
            checkpoint(update+1,"checkpoint.pt")
            if (update+1)%16 == 0:
                checkpoint(update+1,f"update_{update+1}.pt")
        emit("training_completed",status="trained")
        # Destroy training wrapper before constructing independent original episodes.
        env.close()
        env = None
        if not args.skip_evaluation:
            evaluator = Path(__file__).resolve().with_name("evaluate.py")
            emit("evaluation_dispatch", status="evaluating", evaluation_workers=args.evaluation_workers)
            # Replace the process so optimizer, rollout buffers, graph references
            # and the CUDA allocator pool are all released before evaluation.
            os.execv(sys.executable, [sys.executable, "-u", str(evaluator),
                     "--run-dir", str(out), "--out", str(out/"parallel_evaluation"),
                     "--backend", args.backend, "--workers", str(args.evaluation_workers)])
    except BaseException as error:
        status.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        baseline.write_json(out/"status.json",status)
        raise
    finally:
        if env is not None:
            env.close()


def nn_huber(torch, prediction, target):
    return torch.nn.functional.smooth_l1_loss(prediction,target,beta=1.)


if __name__ == "__main__":
    main()
