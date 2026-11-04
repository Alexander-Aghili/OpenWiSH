//! OpenAI-compatible chat completions: OpenAI, OpenRouter, LM Studio, llama.cpp, vLLM, Ollama and others.
//! Open and local models (for example Qwen through LM Studio, or Kimi through OpenRouter) run here.
//!
//! Configuration: `--model` (required), `OPENWISH_OPENAI_BASE_URL` (default `https://api.openai.com/v1`), and a key in
//! `OPENWISH_OPENAI_API_KEY` or `OPENAI_API_KEY` (optional for local servers). `--effort` is sent as
//! `reasoning_effort` unless `OPENWISH_OPENAI_NO_EFFORT=1`, for servers that refuse unknown fields.
use crate::{classify, BackendConfig, BackendError, Editor, Reply, Usage};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAiCompat {
    cfg: BackendConfig,
    base_url: String,
    api_key: Option<String>,
    messages: Vec<Value>,
    session: Option<String>,
}

impl OpenAiCompat {
    pub fn new(cfg: BackendConfig) -> anyhow::Result<Self> {
        if cfg.model.as_deref().unwrap_or("").is_empty() {
            anyhow::bail!("the openai backend needs --model (the model name your server or provider uses)");
        }
        let base_url = std::env::var("OPENWISH_OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let api_key = std::env::var("OPENWISH_OPENAI_API_KEY").or_else(|_| std::env::var("OPENAI_API_KEY")).ok().filter(|k| !k.is_empty());
        Ok(OpenAiCompat { cfg, base_url, api_key, messages: vec![], session: None })
    }

    pub fn body(&self) -> Value {
        let mut messages = vec![json!({"role": "system", "content": self.cfg.instructions})];
        messages.extend(self.messages.iter().cloned());
        let mut b = json!({
            "model": self.cfg.model.clone().unwrap_or_default(),
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
        });
        let no_effort = std::env::var("OPENWISH_OPENAI_NO_EFFORT").map(|v| v == "1").unwrap_or(false);
        if let (Some(e), false) = (&self.cfg.effort, no_effort) {
            b["reasoning_effort"] = json!(e);
        }
        b
    }

    fn post(&self) -> Result<(u16, Value), BackendError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(self.cfg.timeout_s))
            .build()
            .map_err(|e| BackendError::Fatal(e.to_string()))?;
        let mut req = client.post(completions_url(&self.base_url)).header("content-type", "application/json");
        if let Some(k) = &self.api_key {
            req = req.header("authorization", format!("Bearer {k}"));
        }
        let resp = req.json(&self.body()).send().map_err(|e| BackendError::Transient(format!("connection error: {e}")))?;
        let status = resp.status().as_u16();
        let text = resp.text().map_err(|e| BackendError::Transient(format!("connection error reading response: {e}")))?;
        let v = serde_json::from_str(&text).unwrap_or_else(|_| json!({"error": {"message": text.chars().take(2000).collect::<String>()}}));
        Ok((status, v))
    }
}

/// The chat-completions endpoint for a base URL given with or without the path.
pub fn completions_url(base: &str) -> String {
    let b = base.trim_end_matches('/');
    if b.ends_with("/chat/completions") {
        b.to_string()
    } else {
        format!("{b}/chat/completions")
    }
}

/// Parse a chat-completions response (and its HTTP status) into a reply.
pub fn parse_response(status: u16, v: &Value) -> Result<Reply, BackendError> {
    if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
        let msg = e["message"].as_str().map(String::from).unwrap_or_else(|| e.to_string());
        return Err(match status {
            429 | 500..=599 => BackendError::Transient(format!("HTTP {status}: {msg}")),
            _ => match classify(&msg) {
                BackendError::Refused(m) => BackendError::Refused(m),
                _ if status == 0 || status >= 400 => BackendError::Fatal(format!("HTTP {status}: {msg}")),
                other => other,
            },
        });
    }
    if !(200..300).contains(&status) {
        return Err(if status == 429 || status >= 500 { BackendError::Transient(format!("HTTP {status}")) } else { BackendError::Fatal(format!("HTTP {status}")) });
    }
    let choice = &v["choices"][0];
    let finish = choice["finish_reason"].as_str().unwrap_or("");
    if finish == "content_filter" {
        return Err(BackendError::Refused("finish_reason content_filter".into()));
    }
    let text = choice["message"]["content"].as_str().unwrap_or("").to_string();
    if text.trim().is_empty() && finish == "length" {
        return Err(BackendError::Fatal("the reply hit max_tokens before any answer text (raise --max-tokens)".into()));
    }
    let u = &v["usage"];
    let n = |p: &Value| p.as_u64().unwrap_or(0);
    let prompt = n(&u["prompt_tokens"]);
    let cached = n(&u["prompt_tokens_details"]["cached_tokens"]);
    let usage = Usage {
        input: prompt.saturating_sub(cached),
        output: n(&u["completion_tokens"]),
        cache_read: cached,
        cache_write: 0,
        reasoning: n(&u["completion_tokens_details"]["reasoning_tokens"]),
    };
    Ok(Reply {
        text,
        usage,
        context_tokens: None,
        session: v["id"].as_str().map(String::from),
        tool_free: true,
        cost_usd: u["cost"].as_f64(),
        wall_s: 0.0,
        raw: v.clone(),
    })
}

impl Editor for OpenAiCompat {
    fn name(&self) -> String {
        let host = self.base_url.split("://").nth(1).unwrap_or(&self.base_url).split('/').next().unwrap_or("");
        format!("openai:{}@{host}", self.cfg.model.clone().unwrap_or_default())
    }
    fn send(&mut self, msg: &str, new_conversation: bool) -> Result<Reply, BackendError> {
        if new_conversation {
            self.messages.clear();
        }
        self.messages.push(json!({"role": "user", "content": msg}));
        let t0 = Instant::now();
        match self.post().and_then(|(status, v)| parse_response(status, &v)) {
            Ok(mut r) => {
                // only the answer goes back; reasoning text stays in the run's raw record
                self.messages.push(json!({"role": "assistant", "content": r.text}));
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
    use serde_json::json;
    use std::io::{Read, Write};

    fn cfg(model: Option<&str>, effort: Option<&str>) -> BackendConfig {
        BackendConfig {
            kind: "openai".into(),
            model: model.map(String::from),
            effort: effort.map(String::from),
            instructions: "Answer the user's request.".into(),
            tool_blocks: vec![],
            final_blocks: vec![],
            workdir: "/tmp".into(),
            max_tokens: 32000,
            timeout_s: 5,
        }
    }

    fn backend(base: &str, effort: Option<&str>) -> OpenAiCompat {
        OpenAiCompat {
            cfg: cfg(Some("qwen3.8-27b"), effort),
            base_url: base.into(),
            api_key: None,
            messages: vec![],
            session: None,
        }
    }

    #[test]
    fn url_accepts_a_base_with_or_without_the_path() {
        assert_eq!(completions_url("http://localhost:1234/v1"), "http://localhost:1234/v1/chat/completions");
        assert_eq!(completions_url("http://localhost:1234/v1/"), "http://localhost:1234/v1/chat/completions");
        assert_eq!(completions_url("https://openrouter.ai/api/v1/chat/completions"), "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn body_has_system_messages_max_tokens_and_effort_only_when_set() {
        let mut b = backend("http://x/v1", Some("high"));
        b.messages.push(json!({"role": "user", "content": "hi"}));
        let v = b.body();
        assert_eq!(v["model"], "qwen3.8-27b");
        assert_eq!(v["messages"][0], json!({"role": "system", "content": "Answer the user's request."}));
        assert_eq!(v["messages"][1]["content"], "hi");
        assert_eq!(v["max_tokens"], 32000);
        assert_eq!(v["reasoning_effort"], "high");
        assert!(backend("http://x/v1", None).body().get("reasoning_effort").is_none());
    }

    #[test]
    fn parses_text_usage_reasoning_and_errors() {
        let ok = json!({"id": "c1", "choices": [{"message": {"role": "assistant", "content": "Breakout", "reasoning_content": "long"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 30}, "completion_tokens_details": {"reasoning_tokens": 25}}});
        let r = parse_response(200, &ok).unwrap();
        assert_eq!(r.text, "Breakout");
        assert_eq!(r.usage, Usage { input: 70, output: 40, cache_read: 30, cache_write: 0, reasoning: 25 });
        assert_eq!(r.session.as_deref(), Some("c1"));
        assert!(r.tool_free);
        let filtered = json!({"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]});
        assert!(matches!(parse_response(200, &filtered), Err(BackendError::Refused(_))));
        let busy = json!({"error": {"message": "Rate limit exceeded", "code": 429}});
        assert!(matches!(parse_response(429, &busy), Err(BackendError::Transient(_))));
        let down = json!({"error": {"message": "upstream failed"}});
        assert!(matches!(parse_response(502, &down), Err(BackendError::Transient(_))));
        let bad = json!({"error": {"message": "model not found"}});
        assert!(matches!(parse_response(404, &bad), Err(BackendError::Fatal(_))));
        let cut = json!({"choices": [{"message": {"content": null}, "finish_reason": "length"}]});
        assert!(matches!(parse_response(200, &cut), Err(BackendError::Fatal(m)) if m.contains("max")));
    }

    /// One request against a local server: the conversation is sent, the reply is kept for the next turn.
    #[test]
    fn talks_to_a_local_server_and_keeps_the_conversation() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut bodies = vec![];
            for reply in ["first", "second"] {
                let (mut s, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let mut req = String::new();
                loop {
                    let n = s.read(&mut buf).unwrap();
                    req.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if let Some(h) = req.find("\r\n\r\n") {
                        let len: usize = req[..h].lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
                        if req.len() >= h + 4 + len {
                            bodies.push(req[h + 4..h + 4 + len].to_string());
                            break;
                        }
                    }
                }
                let body = json!({"id": "c", "choices": [{"message": {"content": reply}, "finish_reason": "stop"}], "usage": {}}).to_string();
                write!(s, "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body).unwrap();
            }
            bodies
        });
        let mut b = backend(&format!("http://127.0.0.1:{port}/v1"), None);
        assert_eq!(b.send("hello", true).unwrap().text, "first");
        assert_eq!(b.send("again", false).unwrap().text, "second");
        let bodies = server.join().unwrap();
        let second: Value = serde_json::from_str(&bodies[1]).unwrap();
        let roles: Vec<&str> = second["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert_eq!(second["messages"][2]["content"], "first");
    }

    #[test]
    fn a_model_is_required() {
        std::env::set_var("OPENWISH_OPENAI_BASE_URL", "http://localhost:1234/v1");
        assert!(OpenAiCompat::new(cfg(None, None)).is_err());
        assert!(OpenAiCompat::new(cfg(Some("m"), None)).is_ok());
    }
}
