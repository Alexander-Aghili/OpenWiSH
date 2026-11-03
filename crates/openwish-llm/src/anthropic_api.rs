//! Anthropic Messages API (raw HTTP; there is no official Rust SDK). The harness keeps the full
//! message list; assistant content (including thinking blocks) is passed back unchanged.
use crate::{classify, BackendConfig, BackendError, Editor, Reply, Usage, DEFAULT_CLAUDE_MODEL};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

pub struct AnthropicApi {
    cfg: BackendConfig,
    messages: Vec<Value>,
    /// (header, value): an API key, or a bearer token
    auth: (String, String),
    session: Option<String>,
}

impl AnthropicApi {
    pub fn new(cfg: BackendConfig) -> anyhow::Result<Self> {
        let auth = if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
            ("x-api-key".to_string(), k)
        } else if let Ok(t) = std::env::var("ANTHROPIC_AUTH_TOKEN") {
            ("authorization".to_string(), format!("Bearer {t}"))
        } else {
            anyhow::bail!("the api backend needs ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN (or use --backend claude)");
        };
        Ok(AnthropicApi {
            cfg,
            messages: vec![],
            auth,
            session: None,
        })
    }
    pub fn body(&self) -> Value {
        let mut b = json!({
            "model": self.cfg.model.as_deref().unwrap_or(DEFAULT_CLAUDE_MODEL),
            "max_tokens": self.cfg.max_tokens,
            "system": self.cfg.instructions,
            "thinking": {"type": "adaptive"},
            "messages": self.messages,
        });
        if let Some(e) = &self.cfg.effort {
            b["output_config"] = json!({"effort": e});
        }
        b
    }

    /// POST the conversation so far and return the response body.
    fn post(&self) -> Result<Value, BackendError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(self.cfg.timeout_s))
            .build()
            .map_err(|e| BackendError::Fatal(e.to_string()))?;
        let mut req = client
            .post(MESSAGES_URL)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .header(self.auth.0.as_str(), self.auth.1.as_str());
        if self.auth.0 == "authorization" {
            req = req.header("anthropic-beta", "oauth-2025-04-20");
        }
        let resp = req
            .json(&self.body())
            .send()
            .map_err(|e| classify(&format!("connection error: {e}")))?;
        resp.json()
            .map_err(|e| classify(&format!("connection error reading response: {e}")))
    }
}

/// Parse a Messages API response body into the reply and the assistant content to send back next turn.
pub fn parse_response(v: &Value) -> Result<(Reply, Value), BackendError> {
    if v["type"] == "error" {
        let e = &v["error"];
        let (kind, message) = (e["type"].as_str().unwrap_or(""), e["message"].as_str().unwrap_or(""));
        return Err(classify(&format!("{kind} {message}")));
    }
    if v["stop_reason"] == "refusal" {
        let category = v["stop_details"]["category"].as_str().unwrap_or("unknown");
        return Err(BackendError::Refused(format!("stop_reason refusal ({category})")));
    }
    let content = v["content"].clone();
    let text = content
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b["type"] == "text")
                .filter_map(|b| b["text"].as_str())
                .collect()
        })
        .unwrap_or_default();
    let reply = Reply {
        text,
        usage: Usage::from_anthropic(&v["usage"]),
        context_tokens: None,
        session: v["id"].as_str().map(String::from),
        tool_free: true,
        cost_usd: None,
        wall_s: 0.0,
        raw: v.clone(),
    };
    Ok((reply, content))
}

impl Editor for AnthropicApi {
    fn name(&self) -> String {
        format!("anthropic-api:{}", self.cfg.model.clone().unwrap_or_default())
    }
    fn send(&mut self, msg: &str, new_conversation: bool) -> Result<Reply, BackendError> {
        if new_conversation {
            self.messages.clear();
        }
        self.messages.push(json!({"role": "user", "content": msg}));
        let t0 = Instant::now();
        match self.post().and_then(|v| parse_response(&v)) {
            Ok((mut r, content)) => {
                self.messages.push(json!({"role": "assistant", "content": content}));
                r.wall_s = t0.elapsed().as_secs_f64();
                self.session = r.session.clone();
                Ok(r)
            }
            Err(e) => {
                // the harness sends the message again when it retries
                self.messages.pop();
                Err(e)
            }
        }
    }
    fn session(&self) -> Option<String> {
        self.session.clone()
    }
    fn set_session(&mut self, s: Option<String>) {
        self.session = s;
    }
    /// The message list lives in this process only, so a restarted run continues through a harness handoff.
    fn can_resume(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_and_usage_and_refusal() {
        let v = json!({"id":"msg_1","content":[{"type":"thinking","thinking":"","signature":"s"},{"type":"text","text":"hello"}],
            "stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":3,"cache_read_input_tokens":2,"cache_creation_input_tokens":1}});
        let (r, content) = parse_response(&v).unwrap();
        assert_eq!(r.text, "hello");
        assert_eq!(
            r.usage,
            Usage {
                input: 10,
                output: 3,
                cache_read: 2,
                cache_write: 1,
                reasoning: 0
            }
        );
        assert_eq!(content.as_array().unwrap().len(), 2);
        let refused = json!({"content":[],"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber"},"usage":{}});
        assert!(matches!(parse_response(&refused), Err(BackendError::Refused(_))));
        let err = json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}});
        assert!(matches!(parse_response(&err), Err(BackendError::Transient(_))));
    }

    #[test]
    fn body_uses_adaptive_thinking_and_effort() {
        let cfg = BackendConfig {
            kind: "api".into(),
            model: Some("claude-opus-5-5".into()),
            effort: Some("high".into()),
            instructions: "I".into(),
            tool_blocks: vec![],
            final_blocks: vec![],
            workdir: "/tmp".into(),
            max_tokens: 16000,
            timeout_s: 1,
        };
        let a = AnthropicApi {
            cfg,
            messages: vec![json!({"role":"user","content":"x"})],
            auth: ("x-api-key".into(), "k".into()),
            session: None,
        };
        let b = a.body();
        assert_eq!(b["model"], "claude-opus-5-5");
        assert_eq!(b["thinking"]["type"], "adaptive");
        assert_eq!(b["output_config"]["effort"], "high");
        assert_eq!(b["system"], "I");
        assert_eq!(b["max_tokens"], 16000);
    }
}
