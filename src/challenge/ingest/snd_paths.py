from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class SNDPaths:
    root: Path
    def processed(self) -> Path: return self.root / "data" / "processed"
    def house(self) -> Path:     return self.root / "data" / "house_processed"

    # Train
    def train_op(self) -> Path:  return self.processed() / "train_operational_readouts.csv"
    def train_spec(self) -> Path:return self.processed() / "train_specifications.csv"
    def train_tte(self) -> Path: return self.processed() / "train_tte.csv"

    # Validation
    def val_op(self) -> Path:    return self.processed() / "validation_operational_readouts.csv"
    def val_spec(self) -> Path:  return self.processed() / "validation_specifications.csv"
    def val_labels(self) -> Path:return self.processed() / "validation_labels.csv"

    # Test
    def test_op(self) -> Path:   return self.processed() / "test_operational_readouts.csv"
    def test_spec(self) -> Path: return self.processed() / "test_specifications.csv"
