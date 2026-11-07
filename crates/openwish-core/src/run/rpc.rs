//! The kernel's requests to the harness (save_edit, checkpoints, restore), answered through the
//! edit gate.
use super::*;

/// Answer one edit RPC from the sandbox kernel.
pub(super) fn handle_rpc(gate: Option<&mut Gate>, r: Rpc, turn: u32, rpc_dir: &Path) -> RpcReply {
    let Some(g) = gate else {
        return RpcReply {
            ok: false,
            msg: "this task has no editable network".into(),
        };
    };
    let src = format!("python (turn {turn})");
    let answer = match r {
        // only arrays the kernel itself wrote into /work/.rpc are accepted
        Rpc::SaveEdit { name, path, reason } => match read_rpc_array(rpc_dir, &path) {
            Ok(t) => edit_answer(g.replace(&name, t, &reason, &src)),
            Err(e) => Err(format!("could not read the array: {e}")),
        },
        Rpc::DropEdit { name, reason } => edit_answer(g.apply(EditOp::Revert { tensor: name }, &reason, &src)),
        Rpc::Checkpoint { name } => g
            .checkpoint(&name)
            .map(|()| format!("checkpoint {name} saved"))
            .map_err(|e| e.to_string()),
        Rpc::Restore { name } => g
            .restore(&name, &src)
            .map(|v| format!("restored checkpoint {name} ({} tensors changed)", v.len()))
            .map_err(|e| e.to_string()),
        Rpc::Ledger => Ok(g.summary()),
    };
    match answer {
        Ok(msg) => RpcReply { ok: true, msg },
        Err(msg) => RpcReply { ok: false, msg },
    }
}

pub(super) fn read_rpc_array(rpc_dir: &Path, path: &Path) -> Result<openwish_tensor::Tensor> {
    let p = paths::file_under(rpc_dir, path)?;
    openwish_tensor::npy::read_npy(&std::fs::read(p)?)
}

pub(super) fn edit_answer(res: std::result::Result<EditRecord, Refusal>) -> std::result::Result<String, String> {
    res.map(|rec| {
        format!(
            "edit #{} saved: {} entries of {} changed; {} entries ({:.4}%) of the network now differ from the original",
            rec.id,
            rec.changed,
            rec.op.tensor(),
            rec.changed_vs_base,
            rec.frac_changed_vs_base * 100.0
        )
    })
    .map_err(|e| e.to_string())
}
