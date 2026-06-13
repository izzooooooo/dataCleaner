"""
Transformation Pipeline — undo/redo history, immutable original snapshot
"""
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransformStep:
    label: str
    df: pd.DataFrame

    def copy(self) -> "TransformStep":
        return TransformStep(label=self.label, df=self.df.copy())


class Pipeline:
    def __init__(self, df: pd.DataFrame):
        self._original = df.copy()
        self._history: list[TransformStep] = [TransformStep("Orijinal", df.copy())]
        self._redo_stack: list[TransformStep] = []

    #  current state 
    @property
    def current(self) -> pd.DataFrame:
        return self._history[-1].df.copy()

    @property
    def original(self) -> pd.DataFrame:
        return self._original.copy()

    #  history 
    @property
    def steps(self) -> list[str]:
        return [s.label for s in self._history]

    #  push 
    def push(self, label: str, df: pd.DataFrame) -> None:
        self._history.append(TransformStep(label, df.copy()))
        self._redo_stack.clear()

    # undo 
    def undo(self) -> Optional[str]:
        if len(self._history) <= 1:
            return None
        popped = self._history.pop()
        self._redo_stack.append(popped)
        return self._history[-1].label

    #  redo 
    def redo(self) -> Optional[str]:
        if not self._redo_stack:
            return None
        step = self._redo_stack.pop()
        self._history.append(step)
        return step.label

    #  revert 
    def revert(self) -> None:
        self._history = [TransformStep("Orijinal", self._original.copy())]
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self._history) > 1

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0
