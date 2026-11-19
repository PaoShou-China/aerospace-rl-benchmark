# Phy-AI offline touchdown prototype (2026-10-29)

Distinct reusable offline prototype, not a trained controller or benchmark improvement.

Compared with public physical_belief_rocket_landing (12 effective parameters, recurrent identification and direct action refinement) and observable_survival_rocket (observable online belief/survival-aware training smoke), this adds an explicit 61-parameter + 6-disturbance + state full 87D joint-covariance contract, four neural intent nodes, short-prefix full-physics shooting, action-dependent uncertainty projection, and command/snapshot binding. Public comparison used origin/Yxuan-Gong=a6b1973 and origin/experiment/observable-survival-rocket-v1=aba6a95 local tracked snapshots.

Local structural evidence: 25/25 intent coordinates altered current actions; changing physical belief with fixed intent altered action; neural gradient flows while factual parameters stay detached. Historical 128-episode replay found cross-contact angular-rate prediction p95 about 0.253 rad/s. Four-candidate CPU timing about 1.44 s. Contact-model G0 remains unmet; no new environment interaction, trained optimizer updates, landing success rate, or real-time guarantee. Frozen benchmark target and short 25ms executable prefix are explicit prototype limitations, not configurable modern-main environment support.

Only dependency-closed frozen physics core, new phyai code, tests, scripts, numeric aggregate evidence, reference calibration parameters and mathematical erratum are included. No weights, raw trajectories, full historical reports, endpoints or personal paths. Replay scripts need a separately authorized original dataset to reproduce replay numbers. Source files unchanged; reports remove source path and per-event arrays only.

Integrate under isolated experimental namespace, not by replacing current main controllers. Run tests in isolation.
