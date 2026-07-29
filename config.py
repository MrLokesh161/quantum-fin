"""Central configuration for the Phase 1 market-collapse pipeline."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path = Path(__file__).resolve().parent
    dataset_dir: Path = project_root / "datasets"
    output_dir: Path = project_root / "outputs"
    visualizations_dir: Path = project_root / "visualizations"
    primary_market: str = "sp500"  # CSV stem used for target and price indicators
    test_size: float = 0.20
    window_size: int = 20
    forecast_horizon: int = 10
    crash_threshold: float = -0.05  # 5% or larger future fall = high risk
    random_state: int = 42
    scaler: str = "standard"  # "standard" or "minmax"

    @property
    def cleaned_path(self) -> Path:
        return self.output_dir / "cleaned_dataset.csv"

    @property
    def engineered_path(self) -> Path:
        return self.output_dir / "engineered_dataset.csv"
