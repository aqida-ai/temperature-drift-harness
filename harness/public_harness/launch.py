"""
AQIDA_CONSTITUTIONAL_SCOPE: gate
THEORY_BINDING_MAP:
- CORE_LOGIC: resource status does not authorize a scientific claim.
- elm_eulerian: keep supervisor ownership separate from the CUDA worker.
- n_s_u: OS lock, process handles and resource records back execution discipline.
CLAIM_BOUNDARY: one owned worker, no external process termination or publication.
FALSIFICATION_TESTS: exclusive lock, sustained idle CUDA before launch, exact
precommitment, and terminate only the owned worker if foreign CUDA appears.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import precommitment as P


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def gpu_pids():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return [int(line.strip()) for line in raw.splitlines() if line.strip()]


@contextmanager
def exclusive_run_lock(path):
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def supervise(command, cwd, output, idle_seconds=60., poll_seconds=15., pid_reader=gpu_pids):
    """Own exactly one subprocess. Never signal a PID returned by pid_reader."""
    if idle_seconds < 0 or poll_seconds <= 0:
        raise ValueError('Invalid supervision interval.')
    output.mkdir(parents=True, exist_ok=True)
    folder = output / ('launch_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + str(os.getpid()))
    folder.mkdir()
    log_path = folder / 'run.log'
    record_path = folder / 'supervisor.json'
    worker = None
    record = dict(status='WAITING_FOR_IDLE_CUDA', supervisor_pid=os.getpid(),
                  started_utc=utc(), idle_seconds=idle_seconds, poll_seconds=poll_seconds,
                  command=command, scientific_verdict_authorized=False)

    def save():
        temporary = record_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        os.replace(temporary, record_path)

    with exclusive_run_lock(output / 'run.lock'):
        save()
        idle_since = None
        while True:
            pids = pid_reader()
            if pids:
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            if idle_since is not None and time.monotonic() - idle_since >= idle_seconds:
                break
            if (output / 'CANCEL_BEFORE_LAUNCH').exists():
                record.update(status='CANCELLED_BEFORE_LAUNCH', finished_utc=utc())
                save()
                return record
            time.sleep(poll_seconds)
        environment = dict(os.environ, PYTHONPATH=str(cwd), PYTHONUNBUFFERED='1')
        try:
            with log_path.open('w', encoding='utf-8') as log_stream, (folder / 'resources.jsonl').open('w', encoding='utf-8') as resources:
                worker = subprocess.Popen(command, cwd=cwd, env=environment,
                                          stdout=log_stream, stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                record.update(status='RUNNING', worker_pid=worker.pid, launched_utc=utc())
                save()
                while True:
                    code = worker.poll()
                    pids = pid_reader()
                    resources.write(json.dumps(dict(utc=utc(), worker_pid=worker.pid, exit_code=code,
                                                    CUDA_pids=pids, free_disk_bytes=shutil.disk_usage(output).free)) + '\n')
                    resources.flush()
                    if code is not None:
                        record.update(status='WORKER_EXITED', worker_exit_code=code, finished_utc=utc())
                        save()
                        return record
                    if any(pid != worker.pid for pid in pids):
                        record.update(status='FOREIGN_CUDA_PROCESS_DETECTED', foreign_CUDA_pids=[pid for pid in pids if pid != worker.pid])
                        worker.terminate()  # Only the Popen handle this supervisor created.
                        try:
                            worker.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.wait(timeout=10)
                        record.update(worker_exit_code=worker.returncode, finished_utc=utc())
                        save()
                        return record
                    time.sleep(poll_seconds)
        except BaseException as error:
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=10)
            record.update(status='SUPERVISOR_FAILED', error_type=type(error).__name__,
                          error=str(error), finished_utc=utc())
            save()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--precommitment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-packages', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    P.verify(root, args.precommitment, require_committed=True)
    record = json.loads(args.precommitment.read_text(encoding='utf-8'))
    own_path = Path(__file__).resolve().relative_to(root).as_posix()
    P.require(own_path in record['source_files'], 'The supervisor itself must be frozen.')
    command = [sys.executable, '-u', '-m', 'public_harness.run', '--root', str(root),
               '--precommitment', str(args.precommitment.resolve()), '--output', str(args.output.resolve())]
    if args.runtime_packages:
        command += ['--runtime-packages', str(args.runtime_packages.resolve())]
    # Always 60 seconds idle and 15-second observations in production. The CPU
    # fixture calls supervise directly with a fake PID reader and shorter waits.
    result = supervise(command, root, args.output.resolve())
    print(json.dumps(result, indent=2))
    if result.get('worker_exit_code', 0) != 0 or result['status'] in {'FOREIGN_CUDA_PROCESS_DETECTED', 'SUPERVISOR_FAILED'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
