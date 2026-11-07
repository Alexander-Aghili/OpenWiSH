use super::*;
use openwish_tensor::{Network, Tensor};
use std::path::Path;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn kernel(d: &Path, timeout_s: u64) -> Option<Kernel> {
    let venv = root().join("assets/venvs/sandbox");
    if !venv.join("bin/python").exists() {
        eprintln!("skipping: no sandbox venv");
        return None;
    }
    let net = d.join("net");
    let mut n = Network::default();
    n.tensors
        .insert("L1.W".into(), Tensor::new(vec![2, 2], vec![1., 2., 3., 4.]));
    n.save_dir(&net).unwrap();
    Some(
        Kernel::start(KernelCfg {
            venv,
            kernel_dir: root().join("python/kernel"),
            private_dir: d.join("private"),
            net_dir: Some(net),
            work_dir: d.join("work"),
            mounts: vec![],
            timeout_s,
            mem_bytes: 8 << 30,
            max_out: 2000,
            prelude: Some("PRELUDE_OK = 41 + 1".into()),
            threads: 2,
        })
        .unwrap(),
    )
}
fn none(_: Rpc) -> RpcReply {
    RpcReply {
        ok: false,
        msg: "no rpc in this test".into(),
    }
}

#[test]
fn variables_persist_and_last_expression_echoes() {
    let d = tempfile::tempdir().unwrap();
    let Some(mut k) = kernel(d.path(), 60) else { return };
    assert_eq!(k.run("x = PRELUDE_OK", &mut none).unwrap().text, "");
    let o = k.run("x * 2", &mut none).unwrap();
    assert_eq!(o.text.trim(), "84");
    let o = k.run("print(load('L1.W').sum(), NAMES)", &mut none).unwrap();
    assert_eq!(o.text.trim(), "10.0 ['L1.W']");
}

#[test]
fn gradients_and_optimisers_raise() {
    let d = tempfile::tempdir().unwrap();
    let Some(mut k) = kernel(d.path(), 120) else { return };
    for code in [
        "import torch\nt = torch.ones(2)\nt.requires_grad_(True)",
        "import torch\ntorch.optim.Adam([torch.zeros(1)])",
        "import torch\nwith torch.enable_grad():\n    pass",
        "import torch\ntorch.autograd.grad",
    ] {
        let o = k.run(code, &mut none).unwrap();
        if code.ends_with("grad") {
            // merely referencing is fine; calling raises
            let o = k
                .run("import torch\ntorch.autograd.grad(None, None)", &mut none)
                .unwrap();
            assert!(o.error && o.text.contains("OpenWiSHForbidden"), "{}", o.text);
            continue;
        }
        assert!(o.error && o.text.contains("OpenWiSHForbidden"), "{code}: {}", o.text);
    }
}

#[test]
fn save_edit_goes_through_rpc() {
    let d = tempfile::tempdir().unwrap();
    let Some(mut k) = kernel(d.path(), 60) else { return };
    let mut got = vec![];
    let mut h = |r: Rpc| {
        if let Rpc::SaveEdit { path, .. } = &r {
            let t = openwish_tensor::npy::read_npy(&std::fs::read(path).unwrap()).unwrap();
            got.push((r.clone(), t.data.to_vec()));
        }
        RpcReply {
            ok: true,
            msg: "edit #1 saved".into(),
        }
    };
    let o = k
        .run(
            "import numpy as np\nsave_edit('L1.W', np.zeros((2,2)), reason='why')",
            &mut h,
        )
        .unwrap();
    assert!(!o.error, "{}", o.text);
    assert!(o.text.contains("edit #1 saved"));
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].1, vec![0.0; 4]);
    let mut refuse = |_: Rpc| RpcReply {
        ok: false,
        msg: "edit refused: nope".into(),
    };
    let o = k.run("save_edit('L1.W', np.ones((2,2)))", &mut refuse).unwrap();
    assert!(o.error && o.text.contains("nope"));
    // weights are read-only inside the sandbox
    let o = k.run("open('/net/L1.W.npy','wb')", &mut none).unwrap();
    assert!(o.error, "{}", o.text);
}

#[test]
fn kernel_restart_keeps_ledger() {
    let d = tempfile::tempdir().unwrap();
    let Some(mut k) = kernel(d.path(), 3) else { return };
    k.run("y = 5", &mut none).unwrap();
    // an edit made on the host side (the ledger) before the crash
    let mut n = Network::load_dir(&d.path().join("net")).unwrap();
    n.tensors.get_mut("L1.W").unwrap().data[0] = 100.0;
    n.save_dir(&d.path().join("net")).unwrap();
    let o = k.run("import time\ntime.sleep(30)", &mut none).unwrap();
    assert!(o.restarted && o.text.contains("restarted"), "{}", o.text);
    let o = k.run("y", &mut none).unwrap();
    assert!(o.error && o.text.contains("NameError"), "{}", o.text);
    let o = k.run("float(load('L1.W')[0,0])", &mut none).unwrap();
    assert_eq!(o.text.trim(), "100.0");
}

#[test]
fn kernel_that_dies_between_runs_is_restarted() {
    let d = tempfile::tempdir().unwrap();
    let Some(mut k) = kernel(d.path(), 30) else { return };
    k.run(
        "import os, threading\nthreading.Timer(0.3, lambda: os._exit(1)).start()",
        &mut none,
    )
    .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1500));
    let o = k.run("1 + 1", &mut none).unwrap();
    assert!(
        o.restarted && o.text.contains("restarted") && o.text.contains('2'),
        "{}",
        o.text
    );
}

#[test]
fn everyday_grad_spellings_are_blocked() {
    let d = tempfile::tempdir().unwrap();
    let Some(mut k) = kernel(d.path(), 120) else { return };
    for code in [
        "import torch\nwith torch.autograd.enable_grad():\n    pass",
        "import torch\ntorch.autograd.set_grad_enabled(True)",
        "import torch\ntorch.autograd.grad_mode.set_grad_enabled(True)",
        "import torch\ntorch._C._set_grad_enabled(True)",
        "import importlib, pkgutil, torch, torch.optim as o\nm = [importlib.import_module('torch.optim.' + i.name) for i in pkgutil.iter_modules(o.__path__) if i.name.strip('_') == 'adam'][0]\nm.Adam([torch.zeros(1)])",
        "import torch\ntorch.autograd.Variable._execution_engine.run_backward",
    ] {
        let o = k.run(code, &mut none).unwrap();
        assert!(o.error && o.text.contains("OpenWiSHForbidden"), "{code}: {}", o.text);
    }
    // with grad mode off, requires_grad tensors build no graph
    let o = k
        .run(
            "import torch\nx = torch.zeros(2, requires_grad=True)\nprint((x * 2).grad_fn is None)",
            &mut none,
        )
        .unwrap();
    assert!(o.text.contains("True"), "{}", o.text);
}

#[test]
fn no_network_and_output_is_cut() {
    let d = tempfile::tempdir().unwrap();
    let Some(mut k) = kernel(d.path(), 60) else { return };
    let o = k
        .run(
            "import socket\ns = socket.create_connection(('1.1.1.1', 80), timeout=3)",
            &mut none,
        )
        .unwrap();
    assert!(o.error, "{}", o.text);
    let o = k.run("print('x' * 10000)", &mut none).unwrap();
    assert!(o.text.len() < 2200 && o.text.contains("cut at"), "{}", o.text.len());
}
