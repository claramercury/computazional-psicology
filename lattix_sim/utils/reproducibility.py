"""
Utilidades de reproducibilidad para la simulación Lattix.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def save_results(results: dict, output_dir: str = "results") -> Path:
    """Serializa resultados numéricos a JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convertir tipos numpy a nativos Python
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    serializable = convert(results)

    filepath = output_path / "summary_stats.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    return filepath
