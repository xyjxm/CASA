# ICRA Figure Style Study for STRIDE Framework

This note records the style references used before regenerating the STRIDE framework with IMAGE2.

## Source Pool

The pool was selected from the public ICRA 2024 accepted paper list:
`https://github.com/ryanbgriffiths/ICRA2024PaperList`

The following 20 accepted-paper titles were inspected through their first-page PDF layout thumbnails, with priority on navigation, safety, legged robots, planning, verification, control, and framework-style robotics papers.

1. Learning Vision-Based Bipedal Locomotion for Challenging Terrain — http://arxiv.org/abs/2309.14594v2
2. Resilient Legged Local Navigation: Learning to Traverse with Compromised Perception End-To-End — http://arxiv.org/abs/2310.03581v1
3. Vision-Language Frontier Maps for Zero-Shot Semantic Navigation — http://arxiv.org/abs/2312.03275v1
4. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration — http://arxiv.org/abs/2310.07896v1
5. Safe Planning in Dynamic Environments Using Conformal Prediction — http://arxiv.org/abs/2210.10254v2
6. Wasserstein Distributionally Robust Chance Constrained Trajectory Optimization for Mobile Robots within Uncertain Safe Corridor — http://arxiv.org/abs/2308.16381v1
7. Distributionally Robust CVaR-Based Safety Filtering for Motion Planning in Uncertain Environments — http://arxiv.org/abs/2309.08821v1
8. Safe POMDP Online Planning Via Shielding — http://arxiv.org/abs/2309.10216v2
9. RoCo: Dialectic Multi-Robot Collaboration with Large Language Models — http://arxiv.org/abs/2307.04738v1
10. Collision Avoidance and Navigation for a Quadrotor Swarm Using End-To-End Deep Reinforcement Learning — http://arxiv.org/abs/2309.13285v2
11. Learning Continuous Control with Geometric Regularity from Robot Intrinsic Symmetry — http://arxiv.org/abs/2306.16316v2
12. Safe Networked Robotics with Probabilistic Verification — http://arxiv.org/abs/2302.09182v4
13. TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers — http://arxiv.org/abs/2310.16985v4
14. MORALS: Analysis of High-Dimensional Robot Controllers Via Topological Tools in a Latent Space — http://arxiv.org/abs/2310.03246v2
15. Under Pressure: Learning-Based Analog Gauge Reading in the Wild — http://arxiv.org/abs/2404.08785v1
16. Monte Carlo Planning in Hybrid Belief POMDPs — http://arxiv.org/abs/2211.07735v2
17. Data Association Aware POMDP Planning with Hypothesis Pruning Performance Guarantees — http://arxiv.org/abs/2303.02139v3
18. Online Modifications for Event-Based Signal Temporal Logic Specifications — http://arxiv.org/abs/2303.18160v1
19. Sampling-Based Reactive Synthesis for Nondeterministic Hybrid Systems — http://arxiv.org/abs/2304.06876v3
20. Unraveling the Single Tangent Space Fallacy: An Analysis and Clarification for Applying Riemannian Geometry in Robot Learning — http://arxiv.org/abs/2310.07902v3

## Observed Figure Style Patterns

- Fig. 1 is usually compact and information-dense, not a large presentation slide.
- Strong papers use 2-4 labeled panels, often `(a)`, `(b)`, `(c)`, rather than one long chain of boxes.
- The most effective framework figures mix a pipeline with a small scene/robot inset or state-space inset.
- Color is restrained: one primary color for the nominal path, one warning color for failure/safety, one green/teal color for recovery or success.
- Arrows are few, thick enough to read, and tied to semantic branches rather than every possible data dependency.
- Boxes are light, flat, and thinly outlined. Heavy shadows, gradients, and oversized titles look non-paper-like.
- Text labels are short noun phrases. Long explanatory sentences are left to the caption.
- Good figures leave whitespace around modules and avoid dense legends when line colors already communicate meaning.
- The method contribution should be visually central, while policy inputs and simulator/evaluator outputs remain peripheral.
- A robotics framework figure benefits from a small spatial inset showing the robot, obstacle/unsafe region, and resumed task target.

## STRIDE Prompt Implications

- Remove the big presentation-style title inside the figure.
- Use panel labels: perception/policy, STRIDE gate, task-return recovery, execution/evaluation.
- Keep exact text sparse to reduce IMAGE2 text errors.
- Show the blocked path and recovery-return path as the key visual novelty.
- Use a small abstract top-down navigation inset rather than fake real camera screenshots.
- Preserve a white, IEEE-like page background and avoid decorative gradients or icons.
