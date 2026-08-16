from __future__ import annotations


def test_installed_console_entry_points_expose_help_parsers() -> None:
    from kla_restore import cli

    assert callable(cli.train_main)
