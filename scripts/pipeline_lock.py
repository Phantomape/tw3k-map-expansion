"""The four pipeline lock slots under <main checkout>\\work (one helper for every script that checks or takes them).

  slot 1  .pipeline-lock      full pipeline (takes 1+2)
  slot 2  .pipeline-lock-2    full pipeline
  slot 3  .pipeline-lock-3    heavy render / verify; 3 and 4 count as one shared heavy slot
  slot 4  .pipeline-lock-4    small jobs (1-4 GB)

Rules (pipeline-lock memory, coordinator 2026-10-30):
- every slot is checked on its own path; never a combined name. A directory named
  '.pipeline-lock,.pipeline-lock-2' (one mkdir with a comma list, 2026-10-29) holds no slot, so held()
  would read the machine as free; stray() reports such names so callers can refuse.
- one plain mkdir per slot (not makedirs), owner token written, re-read after SETTLE_S; release only if the
  token is still ours.
- a slot whose owner process is gone (killed before its release ran: a TaskStop on slot4_run left slot 4,
  2026-11-01) is reported as stale with its owner.txt, once per process, and never removed here: whoever owns
  the lock rules checks it and rmdirs it by hand.
"""
import tw3k_paths
import os
from pathlib import Path
import re
import sys
import time
import uuid

WORK = tw3k_paths.LOCKS
SLOTS = {1: '.pipeline-lock', 2: '.pipeline-lock-2', 3: '.pipeline-lock-3', 4: '.pipeline-lock-4'}
PIPELINE = (1, 2)
HEAVY = (3, 4)
SETTLE_S = 3
OWNER = 'owner.txt'


def path(n, root=WORK):
    return Path(root) / SLOTS[n]


def held(root=WORK, slots=SLOTS):
    """Slot numbers whose directory exists, each checked on its own path."""
    return [n for n in slots if path(n, root).exists()]


def stray(root=WORK):
    """Lock-like entries in root that are not one of the four slot names (comma lists, typos, '-1')."""
    names = set(SLOTS.values())
    return sorted(p.name for p in Path(root).glob('.pipeline-lock*') if p.name not in names)


def busy(root=WORK, slots=SLOTS):
    """Held slots among `slots` plus any stray lock names; [] means free."""
    return [SLOTS[n] for n in held(root, slots)] + stray(root)


def owner(n, root=WORK):
    try:
        return (path(n, root) / OWNER).read_text(encoding='utf-8').strip()
    except OSError:
        return None


def acquire(n, who, root=WORK, settle=SETTLE_S):
    """Take slot n with one mkdir; returns the owner token, or None if the slot is taken or was lost."""
    p = path(n, root)
    try:
        os.mkdir(p)
    except FileExistsError:
        warn_stale(n, root)
        return None
    token = f'{who} {os.getpid()} {uuid.uuid4().hex[:8]}'
    (p / OWNER).write_text(token, encoding='utf-8')
    time.sleep(settle)
    return token if owner(n, root) == token else None


def stale(n, root=WORK):
    """owner.txt of slot n if its owner process is gone, else None (free, owner alive, or no pid to check)."""
    t = owner(n, root)
    if t is None or not re.fullmatch(r'.+ \d+ [0-9a-f]{8}', t):       # only acquire()'s "who pid hex8" is judged
        return None
    return t if not pid_alive(token_pid(t)) else None


_warned = set()


def warn_stale(n, root=WORK):
    """Print a stale slot n to stderr (once per slot and owner per process); returns its owner.txt or None."""
    t = stale(n, root)
    if t is not None and (str(path(n, root)), t) not in _warned:
        _warned.add((str(path(n, root)), t))
        print(f'[pipeline-lock] 残锁 stale lock {path(n, root)}: owner process gone; owner.txt: {t!r}. '
              'Not removed: check it and rmdir it by the lock rules.', file=sys.stderr, flush=True)
    return t


def release(n, token, root=WORK):
    """Remove slot n only if its owner token is still `token`; returns True if released."""
    p = path(n, root)
    if token is None or owner(n, root) != token:
        return False
    (p / OWNER).unlink()
    p.rmdir()
    return True


def valid_name(name):
    """A slot name is one path component: no separators, commas or whitespace."""
    return bool(re.fullmatch(r'\.pipeline-lock(-[2-4])?', name))


# ------------------------------------------------------------------ the GPU
# One RTX 3070 (8 GB) is shared by polish (gpu_noise, gpu_envelope) and the offline renderer (art_review). Who
# runs device work holds <main checkout>\work\.gpu-lock (its own name: not a pipeline slot, never in busy()/stray()), so
# two processes never fill the card at once (coordinator 2026-10-31). Hold it for a unit of work and free the
# device memory (CuPy pools) before releasing: a renderer takes it per shot, polish per process. A lock whose
# owner process is gone is stale and may be taken over.
GPU = '.gpu-lock'


def gpu_path(root=WORK):
    return Path(root) / GPU


def gpu_owner(root=WORK):
    try:
        return (gpu_path(root) / OWNER).read_text(encoding='utf-8').strip()
    except OSError:
        return None


def pid_alive(pid):
    """True while process pid runs (Windows: OpenProcess + exit code; never a signal: os.kill terminates there)."""
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.GetLastError() == 5         # access denied: it exists
    try:
        code = ctypes.c_ulong()
        return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
    finally:
        k.CloseHandle(h)


def token_pid(token):
    try:
        return int(token.split()[-2])
    except (AttributeError, IndexError, ValueError):
        return None


def gpu_stale(root=WORK):
    """True if the GPU lock exists and its owner process is gone (or no owner file long after the mkdir)."""
    p = gpu_path(root)
    if not p.exists():
        return False
    t = gpu_owner(root)
    if t is None:
        try:
            return time.time() - p.stat().st_mtime > SETTLE_S * 2
        except OSError:
            return False
    pid = token_pid(t)
    return pid is not None and not pid_alive(pid)


def gpu_acquire(who, root=WORK, settle=SETTLE_S):
    """Take the GPU with one mkdir (a stale lock is removed first); returns the owner token or None."""
    p = gpu_path(root)
    if gpu_stale(root):
        try:
            (p / OWNER).unlink(missing_ok=True)
            p.rmdir()
        except OSError:
            return None
    try:
        os.mkdir(p)
    except FileExistsError:
        return None
    token = f'{who} {os.getpid()} {uuid.uuid4().hex[:8]}'
    (p / OWNER).write_text(token, encoding='utf-8')
    time.sleep(settle)
    return token if gpu_owner(root) == token else None


def gpu_release(token, root=WORK):
    p = gpu_path(root)
    if token is None or gpu_owner(root) != token:
        return False
    (p / OWNER).unlink()
    p.rmdir()
    return True


class gpu:
    """with gpu('who', wait=600) as tok: ...  tok is None if the GPU stayed busy for `wait` seconds (None: wait for
    ever); the caller then waits again or takes its CPU path. Released on exit."""
    def __init__(self, who, wait=None, root=WORK, poll=2., settle=.5):
        self.who, self.wait, self.root, self.poll, self.settle = who, wait, root, poll, settle
        self.token = None

    def __enter__(self):
        t0 = time.time()
        while True:
            self.token = gpu_acquire(self.who, self.root, self.settle)
            if self.token or (self.wait is not None and time.time() - t0 >= self.wait):
                return self.token
            time.sleep(self.poll)

    def __exit__(self, *exc):
        gpu_release(self.token, self.root)
        self.token = None
