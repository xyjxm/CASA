# Round 2 ICRA-Style Review

## Novelty

The revised draft now makes the task-return recovery contribution clearer. The novelty is still incremental relative to existing safety filters, but the humanoid VLN framing and intent-return behavior are specific enough for a workshop or early conference draft.

## Technical Correctness

The formalism is intentionally light and no longer overclaims safety guarantees. One remaining weakness is that the intent state is described abstractly. A later version should specify the exact state variables and thresholds, but that detail may depend on code that this paper-only task does not modify.

## Experimental Validity

The adapted baseline language is now acceptable. The table consistency mechanism is a strength. The main weakness is statistical: 50 episodes per method supports a draft comparison, but the paper should not overstate significance.

## Clarity

The paper is readable and the tables are compact. The repeated safe-success values in main and compact tables are slightly redundant, but useful because the requested compact table adds timeout and intervention rate.

## Claims

The claims are now appropriately bounded to the local locked artifact set. The limitations paragraph is important and should remain in the final draft.

## Figures

The generated figures are suitable for an initial ICRA-style draft. The overview and recovery schematics are clear. The bar plots should keep visible count labels so reviewers can map rates back to 50 episodes.
