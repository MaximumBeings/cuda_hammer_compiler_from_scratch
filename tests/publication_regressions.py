#!/usr/bin/env python3
"""Negative publication and GPU-runner tests; no CUDA/GPU execution."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import validate_gpu
import verify_site_links


class LinkTests(unittest.TestCase):
    def test_missing_site_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify_site_links.check(tmp)

    def test_fragments_assets_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / 'index.html'
            for body in ('<a href="missing/">bad</a>', '<a href="#absent">bad</a>',
                         '<img src="missing.svg">', '<p id="x"></p><p id="x"></p>',
                         '<a href="../outside.html">escape</a>'):
                index.write_text(body)
                with self.subTest(body=body), self.assertRaises(ValueError):
                    verify_site_links.check(root)
            index.write_text('<p id="x"></p><a href="#x">ok</a>')
            verify_site_links.check(root)


class GPURunnerTests(unittest.TestCase):
    def invoke(self, directory, driver_exit=0, driver_text='numeric checks passed',
               sanitizer_text='ERROR SUMMARY: 0 errors'):
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            output = driver_text if len(cmd) == 1 else 'device info'
            code = driver_exit if len(cmd) == 1 else 0
            if cmd[0] == 'compute-sanitizer':
                output = sanitizer_text
            return subprocess.CompletedProcess(cmd, code, output)
        with patch.object(sys, 'argv', ['validate_gpu.py', '--sanitize', '--log-dir', str(directory)]), \
             patch.object(validate_gpu.shutil, 'which', side_effect=lambda name: '/fake/' + name), \
             patch.object(validate_gpu.subprocess, 'run', side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = validate_gpu.main()
        return status, calls

    def test_failed_or_skipped_driver_cannot_sanitize_stale_child(self):
        for code, text in [(1, 'build failure'), (77, 'no device'), (0, 'CUDA error'), (0, 'MISMATCH')]:
            with self.subTest(code=code, text=text), tempfile.TemporaryDirectory() as tmp:
                status, calls = self.invoke(Path(tmp) / 'run', code, text)
                self.assertEqual(status, 1)
                self.assertFalse(any(cmd[0] == 'compute-sanitizer' for cmd in calls))

    def test_success_requires_all_twelve_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, calls = self.invoke(Path(tmp) / 'run')
            self.assertEqual(status, 0)
            self.assertEqual(sum(cmd[0] == 'compute-sanitizer' for cmd in calls), 12)
        with tempfile.TemporaryDirectory() as tmp:
            status, _ = self.invoke(Path(tmp) / 'run', sanitizer_text='no diagnostic summary')
            self.assertEqual(status, 1)

    def test_existing_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            evidence = path / 'old.log'
            evidence.write_text('immutable evidence')
            with self.assertRaises(SystemExit):
                self.invoke(path)
            self.assertEqual(evidence.read_text(), 'immutable evidence')
            self.assertEqual(list(path.iterdir()), [evidence])


if __name__ == '__main__':
    unittest.main()
