//! Subprocesses that behave like streams.
//!
//! Most remote WebDataset shards are read by shelling out to a transfer tool
//! (`curl`, `gsutil`, `ais`) and consuming its standard output. [`Pipe`] wraps
//! that pattern so callers see an ordinary [`Read`] or [`Write`] and still get
//! an error when the child exits badly.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use webdataset_core::error::{Error, Result};

/// Exit statuses that are never treated as failures.
///
/// 141 is `128 + SIGPIPE`, which a transfer tool reports whenever the reader
/// stops early — routine when only the first few samples of a shard are read.
pub const DEFAULT_IGNORED_STATUS: &[i32] = &[0, 141];

/// A child process presented as a stream.
#[derive(Debug)]
pub struct Pipe {
    description: String,
    child: Child,
    stdout: Option<ChildStdout>,
    stdin: Option<ChildStdin>,
    ignored_status: Vec<i32>,
    ignore_errors: bool,
    status: Option<i32>,
}

impl Pipe {
    /// Spawn `command` and read its standard output.
    pub fn read(command: Command) -> Result<Pipe> {
        Self::spawn(command, false)
    }

    /// Spawn `command` and write to its standard input.
    pub fn write(command: Command) -> Result<Pipe> {
        Self::spawn(command, true)
    }

    fn spawn(mut command: Command, writing: bool) -> Result<Pipe> {
        let description = describe(&command);
        if writing {
            command.stdin(Stdio::piped());
        } else {
            command.stdout(Stdio::piped());
        }
        let mut child =
            command.spawn().map_err(|e| Error::Subprocess(format!("{description}: could not start ({e})")))?;
        let stdout = if writing { None } else { child.stdout.take() };
        let stdin = if writing { child.stdin.take() } else { None };
        Ok(Pipe {
            description,
            child,
            stdout,
            stdin,
            ignored_status: DEFAULT_IGNORED_STATUS.to_vec(),
            ignore_errors: false,
            status: None,
        })
    }

    /// Treat these exit statuses as success, in addition to the defaults.
    pub fn ignore_status(mut self, statuses: &[i32]) -> Pipe {
        self.ignored_status.extend_from_slice(statuses);
        self
    }

    /// Never fail because of the child's exit status.
    pub fn ignore_errors(mut self, ignore: bool) -> Pipe {
        self.ignore_errors = ignore;
        self
    }

    /// How this pipe was started, for error messages.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Close the stream, wait for the child, and report a bad exit status.
    pub fn finish(&mut self) -> Result<()> {
        self.stdout.take();
        self.stdin.take();
        let status = match self.status {
            Some(status) => status,
            None => {
                let status = self.child.wait()?.code().unwrap_or(-1);
                self.status = Some(status);
                status
            }
        };
        log::debug!("{}: exit {status}", self.description);
        if self.ignore_errors || self.ignored_status.contains(&status) {
            return Ok(());
        }
        Err(Error::Subprocess(format!("{}: exit {status}", self.description)))
    }
}

impl Read for Pipe {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let Some(stdout) = self.stdout.as_mut() else {
            return Ok(0);
        };
        let n = stdout.read(buf)?;
        if n == 0 {
            // End of output: the child's status is now meaningful.
            self.finish().map_err(std::io::Error::other)?;
        }
        Ok(n)
    }
}

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self.stdin.as_mut() {
            Some(stdin) => stdin.write(buf),
            None => Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe is closed")),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.stdin.as_mut() {
            Some(stdin) => stdin.flush(),
            None => Ok(()),
        }
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        // Close the handles so the child sees EOF, then reap it. Errors are
        // only logged: a destructor is the wrong place to raise them.
        self.stdout.take();
        self.stdin.take();
        if self.status.is_none() {
            match self.child.wait() {
                Ok(status) => self.status = status.code(),
                Err(e) => log::debug!("{}: could not wait for child ({e})", self.description),
            }
        }
    }
}

/// Render a command the way a shell would show it.
fn describe(command: &Command) -> String {
    let mut out = command.get_program().to_string_lossy().into_owned();
    for arg in command.get_args() {
        out.push(' ');
        out.push_str(&arg.to_string_lossy());
    }
    out
}

/// Build a `Command` that runs `script` through the system shell.
pub fn shell(script: &str) -> Command {
    let mut command = Command::new("/bin/sh");
    command.arg("-c").arg(script);
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_child_output() {
        let mut pipe = Pipe::read(shell("printf hello")).unwrap();
        let mut out = String::new();
        pipe.read_to_string(&mut out).unwrap();
        assert_eq!(out, "hello");
        pipe.finish().unwrap();
    }

    #[test]
    fn reports_a_failing_child() {
        let mut pipe = Pipe::read(shell("exit 3")).unwrap();
        let mut out = Vec::new();
        let err = pipe.read_to_end(&mut out).unwrap_err();
        assert!(err.to_string().contains("exit 3"), "{err}");
    }

    #[test]
    fn honours_ignored_statuses() {
        let mut pipe = Pipe::read(shell("exit 23")).unwrap().ignore_status(&[23]);
        let mut out = Vec::new();
        pipe.read_to_end(&mut out).unwrap();
        pipe.finish().unwrap();
    }

    #[test]
    fn writes_to_a_child() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let mut pipe = Pipe::write(shell(&format!("cat > {}", path.display()))).unwrap();
        pipe.write_all(b"written").unwrap();
        pipe.finish().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "written");
    }

    #[test]
    fn reports_a_missing_program() {
        let err = Pipe::read(Command::new("definitely-not-a-real-program-xyz")).unwrap_err();
        assert!(matches!(err, Error::Subprocess(_)), "{err}");
    }
}
