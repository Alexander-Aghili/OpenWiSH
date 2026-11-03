"""OpenWiSH sandbox kernel: a persistent Python namespace driven over stdin/stdout with JSON lines.

harness -> kernel   {"type": "run", "code": "..."}
kernel  -> harness  {"type": "rpc", "method": "...", ...}     (harness answers {"type": "rpc_reply", ...})
kernel  -> harness  {"type": "done", "output": "...", "error": bool}

User code's prints are captured; only protocol lines reach the real stdout.
Weights are read-only arrays; every change goes back to the harness through save_edit / drop_edit.
"""

import ast
import contextlib
import io
import json
import os
import sys
import traceback

_PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
_PROTO_IN = os.fdopen(os.dup(0), "r", buffering=1)
NET = os.environ.get("OPENWISH_NET_DIR", "/net")
WORK = os.environ.get("OPENWISH_WORK_DIR", "/work")
_RPC_N = 0  # numbers the .npy files save_edit hands to the harness


class OpenWiSHForbidden(RuntimeError):
    """Raised when code tries to use gradients or optimisers, which this task does not allow."""


def _send(obj):
    _PROTO_OUT.write(json.dumps(obj) + "\n")
    _PROTO_OUT.flush()


def _rpc(method, **kw):
    _send({"type": "rpc", "method": method, **kw})
    line = _PROTO_IN.readline()
    if not line:
        raise SystemExit("harness closed the channel")
    return json.loads(line)


def _forbid(what):
    def f(*a, **k):
        raise OpenWiSHForbidden(
            f"{what} is not allowed: this task forbids gradients, autograd and optimisers. "
            "Compute your edit directly and save it with save_edit(name, array)."
        )

    return f


def _install_guards():
    """Switch gradients off in torch and make every way back on raise OpenWiSHForbidden."""
    try:
        import torch
    except ImportError:
        return
    torch.set_grad_enabled(False)
    _forbid_grad_switches(torch)
    _forbid_backward(torch)
    _forbid_optimisers(torch)


class _NoGrad:
    """Stands in for torch.set_grad_enabled: turning gradients off is fine, turning them on raises."""

    def __init__(self, mode=True):
        if mode:
            _forbid("torch.set_grad_enabled(True)")()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __call__(self, fn):
        return fn


def _forbid_grad_switches(torch):
    """Every switch that turns grad mode or requires_grad on, under each of its names."""
    import torch.autograd.grad_mode as grad_mode

    torch.set_grad_enabled = _NoGrad
    torch.enable_grad = _forbid("torch.enable_grad")
    for mod in (torch.autograd, grad_mode):
        mod.enable_grad = _forbid("enable_grad")
        mod.set_grad_enabled = _NoGrad
    real_requires_grad = torch.Tensor.requires_grad_

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            _forbid("requires_grad_(True)")()
        return real_requires_grad(self, False)

    torch.Tensor.requires_grad_ = requires_grad_
    real_c_set = torch._C._set_grad_enabled

    def _c_set_grad_enabled(mode):
        if mode:
            _forbid("torch._C._set_grad_enabled(True)")()
        return real_c_set(False)

    torch._C._set_grad_enabled = _c_set_grad_enabled


def _forbid_backward(torch):
    """Backpropagation, autograd.grad, the torch.func transforms and the autograd engine itself."""
    torch.Tensor.backward = _forbid("Tensor.backward")
    torch.autograd.backward = _forbid("torch.autograd.backward")
    torch.autograd.grad = _forbid("torch.autograd.grad")
    if hasattr(torch, "func"):
        for n in ("grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd", "hessian"):
            if hasattr(torch.func, n):
                setattr(torch.func, n, _forbid(f"torch.func.{n}"))

    class _NoEngine:
        def __getattr__(self, name):
            _forbid("the autograd engine")()

    torch.autograd.Variable._execution_engine = _NoEngine()


def _forbid_optimisers(torch):
    """Optimiser classes in torch.optim and in each of its submodules (torch.optim.adam.Adam, ...)."""
    import importlib
    import pkgutil

    import torch.optim as optim

    base = optim.Optimizer
    mods = [optim]
    for info in pkgutil.iter_modules(optim.__path__):
        try:
            mods.append(importlib.import_module(f"torch.optim.{info.name}"))
        except Exception:
            pass
    for mod in mods:
        for n in dir(mod):
            c = getattr(mod, n)
            if isinstance(c, type) and issubclass(c, base):
                setattr(mod, n, _forbid(f"{mod.__name__}.{n}"))


def _file(name):
    return os.path.join(NET, name.replace("/", "__") + ".npy")


def _names():
    if not os.path.isdir(NET):
        return []
    return sorted(f[:-4].replace("__", "/") for f in os.listdir(NET) if f.endswith(".npy"))


def load(name):
    """The current value of a tensor (your edits included) as a float32 numpy array. It is a copy:
    changing it does not change the network; save_edit does."""
    import numpy as np

    p = _file(name)
    if not os.path.exists(p):
        raise KeyError(f"unknown tensor {name!r}; NAMES lists them")
    return np.load(p).astype(np.float32)


def _checked_rpc(method, refused=None, done=None, **kw):
    """An RPC whose refusal raises ValueError with the harness's message; returns the message."""
    r = _rpc(method, **kw)
    if not r.get("ok"):
        raise ValueError(r.get("msg", refused))
    return r.get("msg", done)


def save_edit(name, array, reason=""):
    """Replace tensor `name` with `array` (same shape). Goes through the harness's edit gate."""
    import numpy as np

    global _RPC_N
    a = np.asarray(array.detach().cpu().numpy() if hasattr(array, "detach") else array, dtype=np.float32)
    os.makedirs(os.path.join(WORK, ".rpc"), exist_ok=True)
    _RPC_N += 1
    p = os.path.join(WORK, ".rpc", f"edit_{os.getpid()}_{_RPC_N}.npy")
    np.save(p, a)
    try:
        return _checked_rpc("save_edit", "edit refused", "saved", name=name, path=p, reason=str(reason))
    finally:
        try:
            os.remove(p)
        except OSError:
            pass


def drop_edit(name, reason=""):
    """Restore tensor `name` to its original value."""
    return _checked_rpc("drop_edit", "refused", "dropped", name=name, reason=str(reason))


def checkpoint(name):
    """Remember the current edit under `name` (restore it later with restore(name))."""
    return _checked_rpc("checkpoint", name=name)


def restore(name):
    """Bring every tensor back to its state at checkpoint `name`."""
    return _checked_rpc("restore", name=name)


def edits():
    """A summary of every edit made so far."""
    return _rpc("ledger").get("msg", "")


NAMES = _names()
NS = {
    "__name__": "__main__",
    "load": load,
    "save_edit": save_edit,
    "drop_edit": drop_edit,
    "checkpoint": checkpoint,
    "restore": restore,
    "edits": edits,
    "NAMES": NAMES,
    "OpenWiSHForbidden": OpenWiSHForbidden,
}


def _run(code):
    buf = io.StringIO()
    err = False
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, mode="exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), NS)
            if last is not None:
                v = eval(compile(last, "<cell>", "eval"), NS)
                if v is not None:
                    print(repr(v))
        except SystemExit:
            print("[SystemExit ignored]")
        except BaseException:
            err = True
            tb = traceback.format_exc()
            # drop the kernel's own frames
            print("\n".join(line for line in tb.splitlines() if "kernel.py" not in line))
    return buf.getvalue(), err


def _print_defaults():
    try:
        import numpy as np

        np.set_printoptions(precision=4, suppress=True, linewidth=160)
    except ImportError:
        pass


def main():
    _print_defaults()
    _install_guards()
    prelude = os.environ.get("OPENWISH_PRELUDE_FILE")
    if prelude and os.path.exists(prelude):
        with open(prelude) as f:
            out, err = _run(f.read())
        if err:
            _send({"type": "done", "output": "[prelude failed]\n" + out, "error": True})
    _send({"type": "ready"})
    for line in _PROTO_IN:
        if not line.strip():
            continue
        msg = json.loads(line)
        if msg.get("type") == "run":
            NS["NAMES"] = _names()
            out, err = _run(msg["code"])
            _send({"type": "done", "output": out, "error": err})


if __name__ == "__main__":
    main()
