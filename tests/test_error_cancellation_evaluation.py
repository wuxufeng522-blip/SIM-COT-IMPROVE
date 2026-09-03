from reliable_simcot.error_cancellation_evaluation import _effect


def test_evaluation_module_imports() -> None:
    assert callable(_effect)
