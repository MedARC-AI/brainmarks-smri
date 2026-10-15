"""Exclude FOMO300K collections that overlap with any of the evaluation datasets.

    uv run python scripts/filter_fomo300k.py

Reads only the root mapping.tsv at the pinned revision (gated: needs HF_TOKEN with access).
"""

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "FOMO-MRI/FOMO300K"
REVISION = "0ea0b15cfcc08a96b7b9fd43d952cec3fdb7cfc7"
OUT = Path(__file__).parent / "fomo300k_filtered_sessions.txt"

# FOMO collection -> the eval datasets it shares subjects with.
EXCLUDED = {
    "PT005_ADHD_200": "adhd200",
    "PT009_BraTS-GEN": "brats2021, ucsf_pdgm, upenn_gbm",
    "PT014_CoRR": "openbhb",
    "PT015_MSD_BrainTumor": "brats2021, ucsf_pdgm, upenn_gbm",
    "PT016_GSP": "openbhb",
    "PT021_IXI": "ixi, openbhb",
    "PT028_OASIS1": "oasis1",
    "PT029_OASIS2": "oasis2",
    "PT030_OpenNeuro/ds000030": "cnp",
    "PT030_OpenNeuro/ds000221": "openbhb",
    "PT030_OpenNeuro/ds002330": "openbhb",
    "PT030_OpenNeuro/ds002345": "openbhb",
    "PT030_OpenNeuro/ds004889": "soop",
}


def main() -> None:
    mapping_file = hf_hub_download(
        REPO_ID,
        "mapping.tsv",
        repo_type="dataset",
        revision=REVISION,
        token=os.environ.get("HF_TOKEN"),
    )
    mapping = pd.read_csv(mapping_file, sep="\t", dtype=str, keep_default_na=False)
    missing = set(EXCLUDED) - set(mapping.dataset)
    assert not missing, f"excluded collections not in mapping.tsv: {missing}"

    kept = mapping[~mapping.dataset.isin(EXCLUDED)]
    zips = kept.dataset + "/" + kept.participant_id + "/" + kept.session_id + ".zip"
    sessions = sorted(set(zips))
    OUT.write_text("\n".join(sessions) + "\n")
    print(f"kept {len(sessions)} sessions, {len(kept)} of {len(mapping)} scans")


if __name__ == "__main__":
    main()
