"""Dependency-light tests for CLI input and energy-grid validation helpers."""
import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'dmcr' / 'deconv_spec.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'_existing_file', '_check_grid'}]
namespace = {'np': np}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)


class ValidationTests(unittest.TestCase):
    def test_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'data.dat').write_text('test')
            self.assertEqual(namespace['_existing_file'](directory, 'data.dat', 'Spectrum'), directory / 'data.dat')
            with self.assertRaisesRegex(FileNotFoundError, 'Spectrum not found'):
                namespace['_existing_file'](directory, 'missing.dat', 'Spectrum')
            with self.assertRaisesRegex(ValueError, 'nonempty filename'):
                namespace['_existing_file'](directory, None, 'Spectrum')

    def test_grid_validation(self):
        check = namespace['_check_grid']
        grid = np.array([1.0, 2.0, 3.0])
        check(grid, None, 'first')
        check(grid.copy(), grid, 'same')
        with self.assertRaisesRegex(ValueError, 'Energy grid differs'):
            check(np.array([1.0, 2.5, 3.0]), grid, 'different')
        with self.assertRaisesRegex(ValueError, 'No energy points'):
            check(np.array([]), grid, 'empty')


if __name__ == '__main__':
    unittest.main()
