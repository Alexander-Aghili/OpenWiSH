//! How much of the machine OpenWiSH may use. By default a quarter of the cores, at lower priority (`config`
//! resolves the flags, OPENWISH_THREADS / OPENWISH_NICE and openwish.toml). Child processes (sandbox kernel,
//! probes, scorers) get the same limit through the usual thread variables.

/// The default thread count: a quarter of `cores`, at least 1.
pub fn default_threads(cores: usize) -> usize {
    (cores / 4).max(1)
}

/// Environment variables that hold child processes to `n` threads (numpy/BLAS, torch, rayon, JAX/XLA).
pub fn child_env(n: usize) -> Vec<(&'static str, String)> {
    let v = n.to_string();
    vec![
        ("OPENWISH_THREADS", v.clone()),
        ("OMP_NUM_THREADS", v.clone()),
        ("MKL_NUM_THREADS", v.clone()),
        ("OPENBLAS_NUM_THREADS", v.clone()),
        ("RAYON_NUM_THREADS", v.clone()),
        (
            "XLA_FLAGS",
            format!("--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={n}"),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quarter_of_the_cores_by_default() {
        assert_eq!(default_threads(22), 5);
        assert_eq!(default_threads(2), 1);
        assert_eq!(default_threads(1), 1);
    }

    #[test]
    fn child_limits_cover_the_common_libraries() {
        let e = child_env(3);
        for k in [
            "OPENWISH_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "RAYON_NUM_THREADS",
        ] {
            assert!(e.iter().any(|(n, v)| *n == k && v == "3"), "{k}");
        }
        assert!(e
            .iter()
            .any(|(n, v)| *n == "XLA_FLAGS" && v.contains("intra_op_parallelism_threads=3")));
    }
}
