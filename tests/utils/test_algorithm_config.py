import pytest

from brats.constants import (
    ADULT_GLIOMA_PRE_AND_POST_TREATMENT_SEGMENTATION_ALGORITHMS,
    META_DIR,
)
from brats.utils.algorithm_config import load_algorithms


@pytest.fixture
def configs():
    return [
        f for f in META_DIR.iterdir() if f.is_file() and f.suffix in [".yml", ".yaml"]
    ]


def test_configs_valid(configs):
    for config in configs:
        try:
            load_algorithms(file_path=config)
        except Exception as e:  # noqa: BLE001 (test validation)
            pytest.fail(f"Failed to load config {config}: {e}")


def test_integrity_rank(configs):
    ordinal_map = {
        "1": "1st",
        "2": "2nd",
        "3": "3rd",
    }
    for config in configs:
        algorithms = load_algorithms(file_path=config)
        for alg_key, alg_data in algorithms.items():
            name_rank = alg_key.split("_")[-1][0]
            assert alg_data.meta.rank == ordinal_map[name_rank], (
                f"Rank mismatch for {alg_key} in {config}"
            )


def test_integrity_year(configs):
    for config in configs:
        algorithms = load_algorithms(file_path=config)
        for alg_key, alg_data in algorithms.items():
            name_year_abr: str = alg_key.split("_")[0][-2:]
            name_year: int = int(f"20{name_year_abr}")

            assert alg_data.meta.year == name_year


def test_integrity_challenge_manuscript(configs):
    """Should be the same for all algorithms within a year"""
    for config in configs:
        algorithms = load_algorithms(file_path=config)
        # group by year
        challenge_manuscripts_by_years = {}
        for alg_data in algorithms.values():
            year = alg_data.meta.year
            if year not in challenge_manuscripts_by_years:
                challenge_manuscripts_by_years[year] = []
            challenge_manuscripts_by_years[year].append(
                alg_data.meta.challenge_manuscript
            )

        for challenge_manuscripts in challenge_manuscripts_by_years.values():
            assert len(set(challenge_manuscripts)) == 1


def test_subject_id_suffix_defaults_to_none(configs):
    """Only the pre & post treatment track encodes a timepoint suffix."""
    for config in configs:
        if config == ADULT_GLIOMA_PRE_AND_POST_TREATMENT_SEGMENTATION_ALGORITHMS:
            continue
        for alg_data in load_algorithms(file_path=config).values():
            assert alg_data.run_args.subject_id_suffix is None


def test_pre_and_post_subject_id_suffix():
    algorithms = load_algorithms(
        file_path=ADULT_GLIOMA_PRE_AND_POST_TREATMENT_SEGMENTATION_ALGORITHMS
    )
    for alg_data in algorithms.values():
        assert "{timepoint}" in alg_data.run_args.input_name_schema
        assert alg_data.run_args.subject_id_suffix == "100"
