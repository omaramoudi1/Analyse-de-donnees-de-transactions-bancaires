from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_csv: Path = root / "data" / "creditcard.csv"
    model_path: Path = root / "models" / "model.joblib"
    outputs_dir: Path = root / "outputs"

@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    threshold: float = 0.30  
