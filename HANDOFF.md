# P08 handoff

**Verified run** `digits-cpu-5701` under `results/`. Six 72-step LoRAs completed, with nonzero update norms and exact state reload. The published 50/50 joint adapter has L2 update 4.4934. Its saved-weight demo regenerated samples. Re-evaluating the saved observations produced an identical summary CSV. The class accuracy of the six arms is 8.75–12.5%, so extending the conditional base's training budget is a distinct follow-up experiment. Fresh GitHub clone and site publication are checked by the release owner.

The fixed configuration is in `configs/experiment.json`. Run `python run.py reproduce --out work/repeat` for a new complete CPU run. The data ships in scikit-learn. Saved observations can be re-tabulated with `python run.py evaluate --run results --out work/rebuilt`. Run `python run.py demo --out work/demo` to generate samples from the published joint-filter adapter.

The six adapters are compared only on this 8×8 structured-caption task. Each holds a fixed 120-record budget. The table separates injected caption errors and unique near-duplicate groups from the independently measured generation scores. To extend this experiment, freeze a new configuration and output directory before changing the number of training seeds, steps or sources.
