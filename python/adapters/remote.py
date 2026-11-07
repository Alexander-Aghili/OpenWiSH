"""Forward a worker call (probe or score) to a GPU host over ssh and rsync, so model inference leaves this machine.

Set OPENWISH_REMOTE to a JSON file: {"host": "root@IP", "port": 22, "key": "~/.ssh/id_rsa", "root": "/root/openwish",
"python": "/usr/bin/python3"}. The host needs the same repo paths under root (the worker, its assets, the base net).
Per run, the host keeps a copy of the run's network: the first call copies the task's base net there, and every call
rsyncs the run's current net with delta transfer, so only changed bytes travel. The worker then runs on the host with
OPENWISH_DEVICE=cuda, and its output folder comes back. The task files and the edit gate do not change."""
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


def wanted(env):
    return bool(env.get("OPENWISH_REMOTE")) and not env.get("OPENWISH_ON_REMOTE")


def remote_net(local_net, root):
    return f"{root}/nets/{hashlib.sha1(os.path.realpath(local_net).encode()).hexdigest()[:16]}"


def _ssh(cfg):
    # keepalives: a dropped connection ends the call within about a minute instead of hanging until the probe's timeout
    return ["ssh", "-i", os.path.expanduser(cfg["key"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-p", str(cfg["port"])]


def forward(worker, args, base, env, run=subprocess.run):
    """Run `worker args` on the remote host. Returns the worker's stdout."""
    cfg = json.loads(Path(env["OPENWISH_REMOTE"]).read_text())
    root, host, ssh = cfg["root"], cfg["host"], _ssh(cfg)
    rsh = " ".join(shlex.quote(x) for x in ssh)

    def call(cmd, check=True):
        r = run(cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {r.stderr.strip()[-400:]}")
        return r

    score = args[0] == "score"
    local_net = env["OPENWISH_FINAL_NET"] if score else env["OPENWISH_NET"]
    rnet = remote_net(local_net, root)
    call(ssh + [host, f"test -d {rnet} || (mkdir -p {root}/nets && cp -r {root}/{base} {rnet})"])
    call(["rsync", "-a", "--inplace", "--no-whole-file", "--delete", "-e", rsh, local_net.rstrip("/") + "/", f"{host}:{rnet}/"])
    tag = hashlib.sha1(f"{rnet}:{os.getpid()}".encode()).hexdigest()[:12]
    renv = {"OPENWISH_ON_REMOTE": "1", "OPENWISH_DEVICE": "cuda", "OPENWISH_THREADS": env.get("OPENWISH_THREADS", "4")}
    if score:
        args = args + ["--net", rnet]
    else:
        rreq, rout = f"{root}/req/{tag}.json", f"{root}/out/{tag}"
        call(ssh + [host, f"mkdir -p {root}/req {rout}"])
        call(["rsync", "-a", "-e", rsh, env["OPENWISH_REQ"], f"{host}:{rreq}"])
        renv.update(OPENWISH_REQ=rreq, OPENWISH_NET=rnet, OPENWISH_OUT=rout)
    cmd = " ".join(f"{k}={shlex.quote(v)}" for k, v in renv.items())
    cmd = f"cd {root} && {cmd} {cfg['python']} -I {worker} " + " ".join(shlex.quote(a) for a in args)
    r = call(ssh + [host, cmd], check=False)
    if not score:
        Path(env["OPENWISH_OUT"]).mkdir(parents=True, exist_ok=True)
        call(["rsync", "-a", "-e", rsh, f"{host}:{rout}/", env["OPENWISH_OUT"].rstrip("/") + "/"])
        call(ssh + [host, f"rm -rf {rout} {rreq}"])
    if r.returncode != 0:
        raise RuntimeError(f"remote worker failed ({r.returncode}): {r.stderr.strip()[-600:]}")
    return r.stdout


def maybe_forward(worker, base, argv, env=None):
    """If OPENWISH_REMOTE asks for it, run this worker call on the GPU host, print its output and return True.
    Otherwise return False and let the worker run here. A failed remote call prints an error and exits 1."""
    env = dict(os.environ if env is None else env)
    if not wanted(env):
        return False
    try:
        print(forward(worker, list(argv), base, env), end="", flush=True)
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        raise SystemExit(1)
    return True
