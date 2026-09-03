from rsr_rd_simcot.config import DataConfig
from rsr_rd_simcot.data import build_dataset, dataset_summary


def test_dataset_is_reproducible_and_has_exact_noise_rate() -> None:
    config = DataConfig(train_size=100, val_size=20, test_size=20, seed=17)
    first = build_dataset(config)
    second = build_dataset(config)

    assert [x.to_dict() for x in first["train"]] == [x.to_dict() for x in second["train"]]
    summary = dataset_summary(first)
    assert summary["train_noisy"] == 20
    assert summary["train_noise_rate"] == 0.2
    assert not any(example.is_noisy for example in first["val"] + first["test"])
    assert summary["step_range"] == {"min": 2, "max": 4}


def test_noise_changes_only_one_step_and_preserves_answer() -> None:
    config = DataConfig(train_size=50, val_size=5, test_size=5, noise_rate=1.0, seed=5)
    train = build_dataset(config)["train"]

    for example in train:
        differences = [
            clean != observed
            for clean, observed in zip(example.clean_steps, example.observed_steps, strict=True)
        ]
        assert sum(differences) == 1
        assert differences[example.noisy_step_index]
        assert example.answer.lstrip("-").isdigit()
