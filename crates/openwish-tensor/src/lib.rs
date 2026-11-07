//! Weight store, edit ops, edit gate, ledger, rendering and the edit language.
pub mod delta;
pub mod editlang;
pub mod editset;
pub mod gate;
pub mod network;
pub mod npy;
pub mod ops;
pub mod render;

pub use network::{Network, Tensor};
