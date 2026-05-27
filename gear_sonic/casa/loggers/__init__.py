"""CASA loggers."""

__all__ = ["SkillExecutionLogger", "RolloutLogger"]


def __getattr__(name):
    if name == "SkillExecutionLogger":
        from .skill_logger import SkillExecutionLogger

        return SkillExecutionLogger
    if name == "RolloutLogger":
        from .rollout_logger import RolloutLogger

        return RolloutLogger
    raise AttributeError(name)
