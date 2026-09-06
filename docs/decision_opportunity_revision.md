> **SUPERSEDED (2026-08-21).** This historical review note is not a source of current claims or counts. Use [`README.md`](../README.md), the `main.tex` in the `leo-conf2-transactions-revision-2026-08-21` submission bundle, and [`results/transactions_evidence/`](../results/transactions_evidence/) as the canonical sources.

# Decision-Opportunity Revision Audit

## Problem Found

Several policies had identical 60-ms success values. The underlying decision
rows were valid, but aggregate success did not reveal whether candidate paths
had different binary outcomes. This made healthy, failed, and genuinely
actionable epochs look equally informative.

## Code Correction

`decision_opportunity.py` now labels every epoch after execution as one of:

- all runtime candidates succeed;
- mixed-outcome decision opportunity;
- all runtime candidates fail;
- only one runtime candidate exists;
- no candidate is currently feasible and the evaluator uses its marked
  emergency fallback.

Future `target_next` values are used only for this post-hoc label. They are not
available to any online policy. The evidence builder exports the regime audit,
opportunity-conditioned policy results, missed-opportunity counts, latency
spread, and pairwise choice agreement.

## Measured Robustness Correction

The exact-horizon COMMECT tail holdout contains 88 closed test epochs. A
separate five-fold expanding-window run adds 289 non-overlapping unseen test
epochs: 59, 58, 57, 57, and 58 by fold. Each fold trains on the past,
calibrates on the immediately preceding block, and tests on the next block.
Boundary samples are removed when their target falls outside the same
partition.

Reactive and evidence-gated success are 0.678 (196/289), while the unadmitted
shield obtains 0.671 (194/289). Temporal, context, and ensemble ranking obtain
0.526 (152/289), 0.595 (172/289), and 0.471 (136/289), respectively. Every
fold has only one independent collection unit, so the gate selects reactive in
all five folds. This negative result is included in the paper and rules out a
measured learned-policy admission claim.

## Interpretation of Equal Results

- Victoria is saturated at 60 ms: only 1 of 648 epochs is actionable.
- The physics-informed session holdout has only 35 actionable epochs among
  4,319 and a 42.8% no-currently-feasible regime.
- Moderate and severe orbital stress contain many mixed candidate sets, but all
  principal policies capture almost every 60-ms opportunity. Their latency
  differences remain measurable, but binary success is not discriminative.
- COMMECT is informative: 72 of 88 fixed-holdout epochs and 236 of 289 rolling
  epochs are mixed-outcome decision opportunities at 60 ms.

## Revised Claim

The implementation supports an auditable, opportunity-aware evidence gate that
defaults to reactive selection when independent evidence is insufficient. The
measured one-drive evaluation demonstrates abstention; it does not establish
learned-policy superiority, universal shift detection, or a next-epoch QoS
guarantee.
