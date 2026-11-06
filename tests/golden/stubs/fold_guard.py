"""Stub of the study's `fold_guard`: the fold scope and its training digest."""
import hashlib
from dataclasses import dataclass


class LeakageError(Exception):
    pass


@dataclass(frozen=True)
class FoldScope:
    fold: int
    train: frozenset
    test: frozenset

    @property
    def train_hash(self):
        return hashlib.sha256(",".join(str(p) for p in sorted(self.train)).encode()).hexdigest()


def fold_scope(assignments, fold):
    test = frozenset(int(p) for p in assignments.loc[assignments["fold"] == fold, "patno"])
    train = frozenset(int(p) for p in assignments["patno"]) - test
    return FoldScope(fold=fold, train=train, test=test)
