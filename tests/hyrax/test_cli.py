from hyrax_cli.main import _normalize_exit_code


def test_normalize_exit_code_treats_none_as_success():
    assert _normalize_exit_code(None) == 0


def test_normalize_exit_code_preserves_integer_exit_code():
    assert _normalize_exit_code(2) == 2


def test_normalize_exit_code_treats_returned_objects_as_success():
    assert _normalize_exit_code(object()) == 0
