#!/usr/bin/env python3
"""Run Chapter 18 drivers; preserve historical evidence and reject failed runs."""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
from validation_manifest import CoverageError, load_manifest, require_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='sm_86')
    p.add_argument('--sanitize', action='store_true')
    p.add_argument('--log-dir', help='new empty output directory; defaults to a dated GPU run')
    a = p.parse_args()
    if not re.fullmatch(r'sm_[0-9]+', a.arch):
        p.error('expected sm_NN')
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        sources = require_inputs(root, 'gpu')
        driver_programs = load_manifest(root)['gpu_drivers']
    except CoverageError as ex:
        p.error('Coverage failure: ' + str(ex))
    for tool in ['nvcc', 'nvidia-smi'] + (['compute-sanitizer'] if a.sanitize else []):
        if not shutil.which(tool):
            p.error(tool + ' unavailable: requested GPU validation cannot run')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    logs = pathlib.Path(a.log_dir) if a.log_dir else root / 'validation/gpu-runs' / stamp
    if not logs.is_absolute():
        logs = root / logs
    if logs.exists() and (not logs.is_dir() or any(logs.iterdir())):
        p.error('log directory must be new or empty; existing evidence will not be overwritten')
    logs.mkdir(parents=True, exist_ok=True)
    records = []

    def run(cmd, name):
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            r = subprocess.run(cmd, cwd=root, text=True, errors='replace',
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        except subprocess.TimeoutExpired as ex:
            output = ex.stdout or ''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            r = subprocess.CompletedProcess(cmd, 124, output + '\nTIMEOUT after 180 seconds\n')
        (logs / name).write_text('COMMAND: ' + repr(cmd) + '\n' + r.stdout + '\nEXIT: ' + str(r.returncode) + '\n')
        records.append({'command': cmd, 'log': name, 'started_utc': started,
                        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'exit_code': r.returncode})
        return r

    info = run(['nvidia-smi'], 'nvidia-smi.txt')
    failed = bool(info.returncode)
    if failed:
        print('GPU unavailable: no runtime validation performed')
    else:
        for src in sources:
            exe = logs / (src.stem + '.bin')
            c = run(['nvcc', '-std=c++17', '-O2', '-arch=' + a.arch,
                     '-DHAMMER_CUDA_ARCH="' + a.arch + '"', str(src), '-o', str(exe)],
                    src.stem + '-compile.txt')
            if c.returncode:
                failed = True
                continue
            r = run([str(exe)], src.stem + '-run.txt')
            if r.returncode or re.search(r'no CUDA.capable|MISMATCH|FAILED|CUDA error', r.stdout, re.I):
                print(src.name, 'runtime validation failed', r.returncode, flush=True)
                failed = True
                # Fixed /tmp child paths may still contain an earlier executable.
                # Never report sanitizer evidence for children of a failed driver.
                continue
            if a.sanitize:
                for program in driver_programs[str(src.relative_to(root))]:
                    for tool in ['memcheck', 'racecheck', 'synccheck']:
                        r = run(['compute-sanitizer', '--tool', tool, '--target-processes',
                                 'all', '--error-exitcode', '99', program],
                                pathlib.Path(program).name + '-' + tool + '.txt')
                        summary_ok = re.search(r'(ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards)', r.stdout) is not None
                        if r.returncode or not summary_ok:
                            print(program, tool, 'failed', r.returncode, 'summary present', summary_ok, flush=True)
                            failed = True
    summary = {'arch': a.arch, 'sanitizers': a.sanitize, 'passed': not failed,
               'sources': {str(src.relative_to(root)): hashlib.sha256(src.read_bytes()).hexdigest()
                           for src in sources}, 'commands': records}
    (logs / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('FAILED' if failed else 'Chapter 18 GPU runtime checks passed; inspect logs for device-specific results.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
