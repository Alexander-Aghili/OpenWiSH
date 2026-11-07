//! Layer 3: static checks on each code run. A rule names a code pattern that suggests training, fitting or
//! running the network; a match is a flag for the auditor and the logs, never a block.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flag {
    pub rule: String,
    pub line: usize,
    pub snippet: String,
    pub why: String,
}

/// A code pattern that suggests training, fitting or running the network.
pub(crate) struct Rule {
    name: &'static str,
    re: &'static str,
    /// the code must also save an edit for the rule to fire
    needs_save: bool,
    why: &'static str,
}

pub(crate) const RULES: &[Rule] = &[
    Rule {
        name: "backward",
        re: r"\.backward\s*\(",
        needs_save: false,
        why: "backpropagation",
    },
    Rule {
        name: "autograd",
        re: r"autograd\.grad|torch\.func\.(grad|vjp|jvp|jacrev|jacfwd)|jax\.(grad|value_and_grad|vjp)|requires_grad\s*=\s*True|requires_grad_\(\s*(True)?\s*\)",
        needs_save: false,
        why: "automatic differentiation",
    },
    Rule {
        name: "optimizer",
        re: r"torch\.optim\.|optax\.|\boptim\.(SGD|Adam|AdamW|LBFGS|RMSprop|Adagrad)\b",
        needs_save: false,
        why: "a gradient optimiser",
    },
    Rule {
        name: "scipy_optimize",
        re: r"scipy\.optimize|from\s+scipy\s+import\s+optimize|\b(minimize|least_squares|curve_fit|differential_evolution|basinhopping)\s*\(",
        needs_save: false,
        why: "numerical optimisation",
    },
    Rule {
        name: "manual_gradient_step",
        re: r"-=\s*\(?\s*(lr|learning_rate|eta|step_size|alpha_lr)\b\s*\*",
        needs_save: false,
        why: "a hand-written gradient step",
    },
    Rule {
        name: "fit_into_weights",
        re: r"\.fit\s*\(",
        needs_save: true,
        why: "a model fitted to data whose result may be written into the weights",
    },
    Rule {
        name: "lstsq_into_weights",
        re: r"lstsq\s*\(|\bpinv\s*\(|\bsolve\s*\(",
        needs_save: true,
        why: "least-squares fitting whose result may be written into the weights",
    },
    Rule {
        name: "forward_pass",
        re: r"(tanh|relu|sigmoid|softmax|gelu|silu|maximum|clip)\s*\([^()]*(\([^()]*\)[^()]*)*@",
        needs_save: false,
        why: "an activation applied to a matrix product (a possible forward pass on inputs)",
    },
    Rule {
        name: "scalar_relu",
        re: r"\bmax\s*\(\s*0(?:\.0)?\s*,",
        needs_save: false,
        why: "a possible scalar ReLU, including component-level network simulation",
    },
];

pub fn known_rule(name: &str) -> bool {
    RULES.iter().any(|rule| rule.name == name)
}

pub(crate) fn compiled() -> &'static Vec<(&'static Rule, regex::Regex)> {
    static C: std::sync::OnceLock<Vec<(&'static Rule, regex::Regex)>> = std::sync::OnceLock::new();
    C.get_or_init(|| RULES.iter().map(|r| (r, regex::Regex::new(r.re).unwrap())).collect())
}

/// Flag code patterns that suggest training, fitting or running the network.
pub fn static_check(code: &str) -> Vec<Flag> {
    let saves = code.contains("save_edit");
    let mut out = vec![];
    for (r, re) in compiled() {
        if r.needs_save && !saves {
            continue;
        }
        for (i, line) in code.lines().enumerate() {
            let l = line.split('#').next().unwrap_or("");
            if re.is_match(l) {
                out.push(Flag {
                    rule: r.name.into(),
                    line: i + 1,
                    snippet: openwish_text::head(line.trim(), 160).to_string(),
                    why: r.why.into(),
                });
                break;
            }
        }
    }
    out
}
