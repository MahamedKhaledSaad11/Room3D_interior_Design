from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

from room3d_worker.client import WorkerClient
from room3d_worker.runner import WorkerBackends, WorkerRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Room3D pull-based worker daemon")
    parser.add_argument(
        "--server-url",
        "--backend-url",
        dest="server_url",
        default="http://127.0.0.1:8000",
        help="Backend control plane URL",
    )
    parser.add_argument("--worker-token", default="dev-worker-token", help="X-Worker-Token secret")
    parser.add_argument("--worker-id", default="room3d-worker", help="Unique worker identifier")
    parser.add_argument("--capabilities", nargs="*", default=[], help="Task types this worker claims")
    parser.add_argument("--interactive-only", action="store_true", help="Claim only interactive SAM tasks")
    parser.add_argument("--work-dir", type=Path, default=None, help="Working directory for stage artifacts")
    parser.add_argument("--storage-dir", type=Path, default=None, help="Optional shared local storage root")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Idle poll interval in seconds")
    parser.add_argument("--heartbeat-interval", type=float, default=15.0, help="Heartbeat interval in seconds")
    parser.add_argument(
        "--isolate-stages",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ROOM3D_ISOLATE_STAGES", "1").strip().lower() not in {"0", "false", "no"},
        help="Run heavy batch stages inside isolated GPU subprocesses",
    )
    parser.add_argument("--once", action="store_true", help="Claim and run at most one task, then exit")
    parser.add_argument("--max-tasks", type=int, default=None, help="Exit after processing N tasks")
    args = parser.parse_args(argv)

    stop = threading.Event()
    warm_thread: threading.Thread | None = None
    if not args.interactive_only and not args.once and args.max_tasks is None and not args.capabilities:
        args.capabilities = [name for name in WorkerClient.BATCH_GPU if name != "pose_benchmark"]

        def interactive_lane() -> None:
            with WorkerClient(args.server_url, args.worker_token) as warm_client:
                warm_runner = WorkerRunner(
                    warm_client, worker_id=f"{args.worker_id}-sam", work_root=args.work_dir,
                    storage_dir=args.storage_dir, capabilities=["segment_init", "segment_refine"],
                    interactive_only=True, poll_interval_sec=min(args.poll_interval, 0.25),
                    heartbeat_interval_sec=args.heartbeat_interval,
                )
                warm_runner.run_forever(stop_event=stop)

        warm_thread = threading.Thread(target=interactive_lane, name="interactive-sam", daemon=True)
        warm_thread.start()

    try:
        return _run(args, stop)
    finally:
        stop.set()
        if warm_thread is not None:
            warm_thread.join(timeout=5.0)


def _run(args: argparse.Namespace, stop: threading.Event) -> int:
    with WorkerClient(args.server_url, args.worker_token) as client:
        runner = WorkerRunner(
            client,
            worker_id=args.worker_id,
            work_root=args.work_dir,
            storage_dir=args.storage_dir,
            capabilities=list(args.capabilities),
            interactive_only=args.interactive_only,
            backends=WorkerBackends(use_isolated_subprocesses=bool(args.isolate_stages)),
            poll_interval_sec=args.poll_interval,
            heartbeat_interval_sec=args.heartbeat_interval,
        )
        if args.once:
            runner.run_once()
        else:
            runner.run_forever(stop_event=stop, max_tasks=args.max_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
