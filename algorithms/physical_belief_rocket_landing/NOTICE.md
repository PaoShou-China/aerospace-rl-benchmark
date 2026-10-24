# Source and provenance

This module packages the physical-belief rocket-control experiment built on [PaoShou-China/aerospace-rl-benchmark](https://github.com/PaoShou-China/aerospace-rl-benchmark). The experiment's benchmark environment is frozen at commit `39e6bb6fd5d0f5432f7405a76946938a469492ce` in `vendor/benchmark_source.tar.gz`. Its original source, model assets, and attribution notices are preserved.

The added work covers causal physical belief, a shared identification/control updater, differentiable flight and contact prediction, PPO over the final hybrid action distribution, and training/evaluation entry points. Dependencies including PyTorch, Genesis, and rsl-rl retain their respective licenses. This module does not assign a new license to the upstream project.

The seven implementation modules in `core/` are byte-for-byte copies of the sources saved with the completed experiment. Integration changes only the portable paths, frozen-environment loading, launchers, and independent evaluation scheduling, as listed in `provenance.json`. All three checkpoints retain their original bytes. Historical paths inside checkpoint metadata are not used to locate runtime dependencies; the launchers use paths within this module.
