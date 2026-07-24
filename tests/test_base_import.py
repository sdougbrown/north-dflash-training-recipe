"""The public base package must remain usable with site packages disabled."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class BaseImportTests(unittest.TestCase):
    def test_base_package_imports_without_optional_torch_or_transformers(self):
        project = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project / "src")
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import north_dflash_training; print('base-import-ok')"],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "base-import-ok")


if __name__ == "__main__":
    unittest.main()
