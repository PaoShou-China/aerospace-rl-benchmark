# Confidence-aware physical inference and sequence control

[中文说明](#中文说明)

## Research objective

Improve the recorded 83.59% rocket-landing policy toward 90% on the same frozen benchmark protocol, while measuring whether factual physics inference actually helps control. The included `artifacts/policy.pt` remains the previously evaluated policy. The architecture described below is a new candidate; its training results are pending.

## Architecture changes

- **Causal reliability:** an exponentially weighted innovation statistic is updated from the previous belief's prediction and the action actually executed, before fitting the newest observation. This prevents a good fit to the current observation from masquerading as predictive confidence. A learned heteroscedastic noise head is trained on actual successor states. The reliability score is not asserted to be a calibrated probability.
- **Broader parameter hypotheses:** paired orthogonal directions replace the single diagonal perturbation. The default uses seven hypotheses across three directions; 25 hypotheses cover all twelve coefficient directions. These remain effective coefficients, with a diagonal scale approximation rather than a full posterior covariance.
- **One masked inference loop:** the same learned coordinate updater, type embeddings, physical objective, projection, and acceptance mechanism are used for parameter identification, predicted-state inference, and action-sequence optimization. Prediction-state inference is trained as an auxiliary conditional solve. Control rollout uses the shared physical transition directly. Parameters are fixed during control; control loss cannot directly rewrite the current factual belief.
- **Sequence control:** the actor optimizes three action knots over twelve 40 Hz control steps, with separate ignition-on/off branches and shared actions across dynamics hypotheses. A shifted previous plan supplies a causal warm start. Only proposals that do not increase the modeled objective are accepted. This is a model-relative condition, not a guarantee about Genesis or real-world safety.
- **Confidence-conditioned authority:** recent prediction accuracy, demonstrated modeled improvement, and distance to landing determine the applied correction. Initial physical authority is increased to 0.7 for this campaign; confidence and improvement still scale its actual action contribution. The final hybrid distribution remains the object optimized by PPO.
- **Full-episode learning:** the reward adapter's success-hold window is set to 1,201 steps, longer than the original 1,200-step episode, so success never causes an extra training reset. Settling bonuses and subsequent failure penalties remain active through the original episode. Evaluation uses the original success and termination criteria.

The default model has 360,915 parameters and a 131-dimensional recurrent memory. A local two-sample CPU forward/backward run passed with finite gradients through the shared updater and prediction-noise head. This is an integration check; task performance is determined by the GPU experiments.

## First campaign

All arms start from the same published actor/critic checkpoint, with fresh optimizers, the same training seed, and 2,097,152 new simulator transitions each. Shared inherited history is recorded separately. Total new interaction budget is 8,388,608.

| Arm | Hypotheses | Horizon / knots | Difference |
|---|---:|---:|---|
| `confidence_sequence` | 7 | 12 / 3 | Full candidate |
| `compact_sequence` | 3 | 8 / 2 | Smaller planning budget |
| `fixed_belief` | 7 | 12 / 3 | Nominal fixed coefficients; no inferred belief in action descriptor |
| `no_physical_solver` | 7 | 12 / 3 | Neural proposal without a physical action correction |

The two planning scales test the accuracy/compute tradeoff; the other two arms isolate online belief and physical action solving. These comparisons do not yet isolate confidence gating, every shared-updater mode, or adaptation to changed true dynamics. Later experiments should target whichever mechanism explains the first results, then confirm a selected candidate on independent seeds.

Two workers share the available GPU. Each has a 22 GiB training-memory budget within a 94% whole-device ceiling. Microbatches start at 4,096 and adapt to actual activation peaks. Each worker runs 2,048 environments; the initial PPO minibatch is 32,768. Each completed arm evaluates four original 128-environment cohorts with seeds `1000007`, `2000007`, `3000007`, and `4000007`.

Reported signals include official event success, terminal settling, post-success failures, normalized prediction error, nominal prediction-interval coverage, confidence bins, accepted physical updates, modeled cost reduction, and the magnitude of applied physical action corrections. Terminal episode records include the effective coefficient estimates and the explanation for the last action. Transition-level diagnostics are correlated samples and are not treated as independent statistical trials.

```bash
python scripts/run.py train --config configs/confidence_sequence.json \
  --out runs/confidence_sequence
python scripts/research_campaign.py --out runs/confidence-study --workers 2
```

Run either a single arm or the campaign. A lock prevents duplicate coordinators. Existing interrupted run directories are retained and marked for explicit continuation rather than overwritten. The campaign stops after its four predefined arms; further architecture changes are based on its results.

## Research connections

The user's physical-intelligence discussion motivates online identification from causal interaction, shared physical operators, factual parameter estimates separated from task optimization, and comparisons under equal interaction/compute budgets. This rocket implementation retains numerical state observations and the fixed rocket structure.

[Dream-MPC (ICML 2026)](https://dream-mpc.github.io/) motivates optimizing a small set of action sequences, reusing plans, and regularizing uncertainty. [TD-MPC2](https://www.tdmpc2.com/) provides a model-based planning reference. [TD-M(PC)² (L4DC 2026)](https://darthutopian.github.io/tdmpc_square/) highlights the mismatch between the planner and the policy being optimized; here PPO scores the actual final action distribution. These are design references, not claims of reproducing those algorithms or inheriting their reported performance.

## 中文说明

本轮目标是让物理信息对控制产生可测量的帮助，并在原有评测口径下向 90% 成功率推进。当前已评估检查点仍为 83.59%；新结构尚不能声称达到目标。

核心改动是：先用上一时刻信念预测已执行动作的后果，再用真实误差更新可信度；辨识、预测与控制复用带变量掩码的更新器；动作从短时保持常数扩展为多节点序列；可信度低时减小物理修正，可信且模型目标确实下降时提高物理作用。预测误差、名义区间覆盖率和实际动作改变量都会记录，不能只看模块是否存在或损失是否下降。

第一轮四组各增加约 210 万次交互，两组并行运行。比较完整结构、轻量规划、固定信念和关闭物理求解。成功后继续按原回合长度训练，评估仍同时报告事件成功与最终停稳。首轮结束后依据结果改架构、选择候选，再用独立种子确认收益。
