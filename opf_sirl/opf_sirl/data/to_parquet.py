"""Convert `Sample`s into the parquet layout used by verl's RLHFDataset (also read by TRL).

Columns: data_source, prompt (chat messages), ability, reward_model{style, ground_truth},
extra_info{index, opf(JSON string)}. Task metadata is kept as one JSON string so the
parquet schema stays flat regardless of the shape of `instance_targets`.
"""
import json
from typing import List

from ..prompts import build_messages
from .schema import Sample


def to_record(sample: Sample, index: int) -> dict:
    return {
        "data_source": f"opf_level{sample.level}",
        "prompt": build_messages(sample),
        "ability": "opf_modeling",
        "reward_model": {"style": "rule", "ground_truth": float(sample.objective)},
        "extra_info": {
            "index": index,
            "opf": json.dumps({
                "id": sample.id,
                "level": sample.level,
                "param_values": sample.param_values,
                "instance_targets": sample.instance_targets,
            }),
        },
    }


def write_parquet(samples: List[Sample], path) -> int:
    import pandas as pd  # only needed when building datasets
    records = [to_record(s, i) for i, s in enumerate(samples) if s.objective is not None]
    pd.DataFrame(records).to_parquet(path)
    return len(records)
