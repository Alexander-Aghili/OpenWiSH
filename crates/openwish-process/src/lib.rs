//! Child processes with a deadline: feed stdin, capture stdout and stderr, and kill the child when it runs too
//! long. The harness runs hooks, forward probes, scorers and the CLI agent backends this way.
use std::io::{self, Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

/// How often `run` checks whether the child has exited.
const POLL: Duration = Duration::from_millis(20);
/// How long to wait for the output pipes once the child is gone, when descendants may still hold them open.
const PIPE_GRACE: Duration = Duration::from_secs(2);

/// How to run one child.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// written to the child's stdin (`None`: stdin is /dev/null)
    pub stdin: Option<String>,
    pub timeout: Duration,
    /// Start the child as the leader of a new process group and kill the whole group at the deadline, so
    /// nothing it started survives. Its output is then awaited only briefly after it exits, because a
    /// process that left the group could keep a pipe open forever.
    pub own_group: bool,
}

/// A child that ran to the end or was killed at the deadline.
#[derive(Clone, Debug)]
pub struct Finished {
    /// `None` when the child was killed at the deadline
    pub status: Option<ExitStatus>,
    pub stdout: String,
    pub stderr: String,
    pub elapsed: Duration,
}

impl Finished {
    pub fn timed_out(&self) -> bool {
        self.status.is_none()
    }

    /// The last `n` characters of stderr.
    pub fn stderr_tail(&self, n: usize) -> &str {
        tail_chars(&self.stderr, n)
    }
}

/// Why a child could not be run.
#[derive(Debug)]
pub enum Failure {
    Start { program: String, source: io::Error },
    Wait { program: String, source: io::Error },
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Start { program, source } => write!(f, "could not start {program}: {source}"),
            Failure::Wait { program, source } => write!(f, "waiting for {program}: {source}"),
        }
    }
}

impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Failure::Start { source, .. } | Failure::Wait { source, .. } => Some(source),
        }
    }
}

/// Run `cmd` (its program, arguments, environment and folder already set) under `opts`.
pub fn run(cmd: &mut Command, opts: Options) -> Result<Finished, Failure> {
    let program = cmd.get_program().to_string_lossy().into_owned();
    cmd.stdin(if opts.stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    })
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    if opts.own_group {
        cmd.process_group(0);
    }
    let started = Instant::now();
    let mut child = cmd.spawn().map_err(|source| Failure::Start {
        program: program.clone(),
        source,
    })?;
    if let Some(input) = opts.stdin {
        feed_stdin(&mut child, input);
    }
    let output = drain_output(&mut child);
    let status = match wait_until(&mut child, started + opts.timeout) {
        Ok(status) => status,
        Err(source) => {
            kill(&mut child, opts.own_group);
            return Err(Failure::Wait { program, source });
        }
    };
    if status.is_none() {
        kill(&mut child, opts.own_group);
    }
    let grace = (opts.own_group || status.is_none()).then_some(PIPE_GRACE);
    let (stdout, stderr) = collect_output(&output, grace);
    Ok(Finished {
        status,
        stdout,
        stderr,
        elapsed: started.elapsed(),
    })
}

/// The last line of `stdout` that starts with `{`: where probes, scorers and `claude -p` put their JSON result.
pub fn last_json_line(stdout: &str) -> Option<&str> {
    stdout.lines().rev().find(|l| l.trim_start().starts_with('{'))
}

/// The last `n` characters of `s`.
pub fn tail_chars(s: &str, n: usize) -> &str {
    if n == 0 {
        return "";
    }
    match s.char_indices().rev().nth(n - 1) {
        Some((i, _)) => &s[i..],
        None => s,
    }
}

/// Write `input` from its own thread, so a large input cannot block the child. A child that does not read
/// its stdin is fine.
fn feed_stdin(child: &mut Child, input: String) {
    let mut stdin = child.stdin.take().expect("stdin is piped");
    std::thread::spawn(move || {
        let _ = stdin.write_all(input.as_bytes());
    });
}

enum Stream {
    Out,
    Err,
}

/// Read stdout and stderr to the end, each on its own thread, so neither pipe can fill up and block the child.
fn drain_output(child: &mut Child) -> Receiver<(Stream, String)> {
    let (tx, rx) = mpsc::channel();
    let pipes: [(Stream, Box<dyn Read + Send>); 2] = [
        (Stream::Out, Box::new(child.stdout.take().expect("stdout is piped"))),
        (Stream::Err, Box::new(child.stderr.take().expect("stderr is piped"))),
    ];
    for (which, mut pipe) in pipes {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut bytes = vec![];
            let _ = pipe.read_to_end(&mut bytes);
            let _ = tx.send((which, String::from_utf8_lossy(&bytes).into_owned()));
        });
    }
    rx
}

/// The exit status, or `None` if the child is still running at `deadline`.
fn wait_until(child: &mut Child, deadline: Instant) -> io::Result<Option<ExitStatus>> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(POLL);
    }
}

fn kill(child: &mut Child, whole_group: bool) {
    if whole_group {
        // SAFETY: killpg on the process group this child leads (`process_group(0)` in `run`)
        unsafe {
            libc::killpg(child.id() as i32, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// stdout and stderr as the reader threads deliver them; with `grace`, give up after that long.
fn collect_output(output: &Receiver<(Stream, String)>, grace: Option<Duration>) -> (String, String) {
    let deadline = grace.map(|g| Instant::now() + g);
    let (mut stdout, mut stderr) = (String::new(), String::new());
    for _ in 0..2 {
        let got = match deadline {
            Some(d) => output.recv_timeout(d.saturating_duration_since(Instant::now())).ok(),
            None => output.recv().ok(),
        };
        match got {
            Some((Stream::Out, s)) => stdout = s,
            Some((Stream::Err, s)) => stderr = s,
            None => break,
        }
    }
    (stdout, stderr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Command {
        let mut c = Command::new("sh");
        c.arg("-c").arg(script);
        c
    }
    fn opts(timeout_s: u64) -> Options {
        Options {
            timeout: Duration::from_secs(timeout_s),
            ..Default::default()
        }
    }

    #[test]
    fn captures_both_streams_the_status_and_stdin() {
        let o = Options {
            stdin: Some("from stdin".into()),
            ..opts(10)
        };
        let f = run(&mut sh("cat; echo oops >&2; exit 3"), o).unwrap();
        assert_eq!((f.stdout.as_str(), f.stderr.as_str()), ("from stdin", "oops\n"));
        assert_eq!(f.status.and_then(|s| s.code()), Some(3));
        assert!(!f.timed_out());
    }

    #[test]
    fn a_child_past_its_deadline_is_killed_with_its_group() {
        let t0 = Instant::now();
        let f = run(
            &mut sh("sleep 30 & sleep 30"),
            Options {
                own_group: true,
                ..opts(1)
            },
        )
        .unwrap();
        assert!(f.timed_out());
        assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", t0.elapsed());
    }

    #[test]
    fn a_missing_program_is_a_start_failure() {
        let e = run(&mut Command::new("/no/such/program"), opts(1)).unwrap_err();
        assert!(matches!(e, Failure::Start { .. }) && e.to_string().starts_with("could not start /no/such/program"));
    }

    #[test]
    fn tail_chars_counts_characters() {
        assert_eq!(tail_chars("héllo", 3), "llo");
        assert_eq!(tail_chars("héllo", 4), "éllo");
        assert_eq!(tail_chars("hi", 10), "hi");
        assert_eq!(tail_chars("hi", 0), "");
    }
}
