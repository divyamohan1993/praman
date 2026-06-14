"""Process-level environment setup for PRAMAN.

Two non-negotiables from the brief:
  1. CPU-only. No CUDA. Verify torch.cuda.is_available() is False at startup.
  2. The runtime verify() path makes NO external network calls (on-prem / air-gap).

Import this module early. ``configure_threads`` is idempotent and safe to call
multiple times; ``assert_cpu_only`` is called on first import.
"""
from __future__ import annotations

import os
import socket
from contextlib import contextmanager

# Number of vCPUs on the target box (AMD Genoa, 6 OCPU = 12 vCPU). Overridable.
_DEFAULT_THREADS = int(os.environ.get("PRAMAN_THREADS", "12"))


def configure_threads(n: int | None = None) -> int:
    """Set BLAS/OpenMP + torch thread counts. Call once at process start."""
    n = int(n or _DEFAULT_THREADS)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        torch.set_num_threads(n)
        # interop threads must be set before any parallel work; guard against re-set.
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
    except Exception:
        pass
    return n


def assert_cpu_only() -> None:
    """Hard-fail if a CUDA build sneaks in. The box has no GPU; a CUDA wheel is a bug."""
    try:
        import torch
        if torch.cuda.is_available():  # pragma: no cover - never true on the box
            raise RuntimeError(
                "CUDA is available but PRAMAN is CPU-only by contract. "
                "Uninstall CUDA torch wheels and reinstall the CPU build."
            )
    except ImportError:
        pass


def set_offline() -> None:
    """Force the HF/runtime stack offline. Used by the air-gapped verify() path."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[var] = "1"


class NetworkBlockedError(RuntimeError):
    """Raised when air-gap mode is on and code attempts an outbound socket."""


@contextmanager
def airgap():
    """Context manager that makes any outbound TCP connection raise.

    This is how we PROVE the verify() path is on-prem without touching the box's
    real networking (which would kill the co-hosted services). The pytest air-gap
    test runs verify() inside this context and asserts it still works.
    """
    set_offline()
    real_socket = socket.socket
    real_create_conn = socket.create_connection

    class _BlockedSocket(real_socket):  # type: ignore[misc, valid-type]
        def connect(self, *a, **k):
            raise NetworkBlockedError("outbound network blocked (air-gap mode)")

        def connect_ex(self, *a, **k):
            raise NetworkBlockedError("outbound network blocked (air-gap mode)")

    def _blocked_create_conn(*a, **k):
        raise NetworkBlockedError("outbound network blocked (air-gap mode)")

    socket.socket = _BlockedSocket  # type: ignore[assignment]
    socket.create_connection = _blocked_create_conn  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = real_socket  # type: ignore[assignment]
        socket.create_connection = real_create_conn  # type: ignore[assignment]


# Apply on import: CPU assertion is cheap and catches the worst misconfig early.
assert_cpu_only()
