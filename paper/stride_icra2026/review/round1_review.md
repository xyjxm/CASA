# Round 1 ICRA-Style Review

## Novelty

The draft has a clear systems contribution: an intent-preserving execution wrapper for humanoid VLN. The idea is plausible, but v0 needs sharper separation from generic safety filtering. The strongest novelty appears to be task-return recovery after rejection, not the existence of a safety gate.

## Technical Correctness

The method description is reasonable but should avoid implying formal safety guarantees. The stop verifier and intent state should be described as execution-time consistency checks rather than oracle access. The draft should also clarify that recovery returns to the instruction intent rather than optimizing a new local objective.

## Experimental Validity

The reported locked comparison is useful, but the baseline framing needs care. The external rows must be described as adapted baselines only. The draft should make clear that the results are from existing local artifacts and not from an official reproduction package.

## Clarity

The paper needs a compact metric definition section before presenting results. The relationship between safe success, unsafe, timeout, intervention rate, and final distance should be explicit.

## Claims

The claims should be narrowed. The paper can claim that STRIDE improves safe success in this artifact set, while preserving zero evaluator-reported unsafe violations. It should not claim general dominance, hardware validation, or formal safety.

## Figures

The figures should state that they are generated schematics or generated plots. If no video frames are used, the caption and manifest should say so directly.
