//! The kernel's requests to the harness: the request types and how a kernel message becomes one.
//! Paths in a request are guest paths under /work; they map to host paths inside the run's work folder.
use std::path::PathBuf;

use crate::Kernel;

#[derive(Clone, Debug, PartialEq)]
pub enum Rpc {
    /// `path` is the host path of the .npy the kernel wrote
    SaveEdit {
        name: String,
        path: PathBuf,
        reason: String,
    },
    DropEdit {
        name: String,
        reason: String,
    },
    Checkpoint {
        name: String,
    },
    Restore {
        name: String,
    },
    Ledger,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RpcReply {
    pub ok: bool,
    pub msg: String,
}

impl Kernel {
    /// The RPC in a protocol line of type "rpc". Paths under /work are mapped to the host's work folder.
    pub(crate) fn rpc_request(&self, v: &serde_json::Value) -> Rpc {
        let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
        match v["method"].as_str().unwrap_or("") {
            "save_edit" => Rpc::SaveEdit {
                name: s("name"),
                path: self.guest_to_host(&s("path")),
                reason: s("reason"),
            },
            "drop_edit" => Rpc::DropEdit {
                name: s("name"),
                reason: s("reason"),
            },
            "checkpoint" => Rpc::Checkpoint { name: s("name") },
            "restore" => Rpc::Restore { name: s("name") },
            _ => Rpc::Ledger,
        }
    }

    pub(crate) fn guest_to_host(&self, p: &str) -> PathBuf {
        match p.strip_prefix("/work/") {
            Some(rest) => self.cfg.work_dir.join(rest),
            None => PathBuf::from(p),
        }
    }
}
