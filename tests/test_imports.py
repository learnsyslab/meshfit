"""Every module must import.

Trivial, and it would have caught a syntax error in `api.py` that the rest of
the suite missed entirely -- nothing else imports the entry point, so the
package was broken while 89 tests passed.
"""

import importlib
import pkgutil

import pytest

import meshfit

MODULES = sorted(m.name for m in pkgutil.iter_modules(meshfit.__path__))

# render/matcher pull bpy and a RoMa checkpoint; import them only when present
OPTIONAL = {"render", "matcher"}

# __main__ runs the CLI on import, by design
SKIP = {"__main__"}


@pytest.mark.parametrize("name", [m for m in MODULES if m not in OPTIONAL | SKIP])
def test_module_imports(name):
    importlib.import_module(f"meshfit.{name}")


@pytest.mark.parametrize("name", sorted(OPTIONAL))
def test_optional_module_imports(name):
    pytest.importorskip("bpy" if name == "render" else "romatch")
    importlib.import_module(f"meshfit.{name}")


def test_cli_help_runs():
    from meshfit.cli import main
    with pytest.raises(SystemExit) as e:
        main(["fit", "--help"])
    assert e.value.code == 0


def test_unassessed_symmetry_is_not_reported_as_confident():
    """`None` must not collapse to `False`: a pose whose orientation was never
    checked is not a pose known to be unambiguous."""

    from meshfit.api import Fit
    from meshfit.pose import Pose

    fit = Fit(pose=Pose.identity())          # no symmetry assessed
    assert fit.symmetry is None
    assert fit.ambiguous is None             # not False
    assert fit.trustworthy is False          # absence of evidence is not evidence
    assert fit.to_dict()["confidence"]["ambiguous"] is None
    assert fit.to_dict()["confidence"]["symmetry_assessed"] is False
