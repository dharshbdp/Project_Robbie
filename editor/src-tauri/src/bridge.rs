use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

static WORKER: Mutex<Option<BridgeWorker>> = Mutex::new(None);

fn project_root() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("ROBBIE_PROJECT_DIR") {
        let root = PathBuf::from(root);
        if root.join("editor/bridge.py").is_file() {
            return Ok(root);
        }
        return Err("ROBBIE_PROJECT_DIR must contain editor/bridge.py".into());
    }
    let mut starts = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))];
    if let Ok(directory) = std::env::current_dir() {
        starts.push(directory);
    }
    if let Ok(executable) = std::env::current_exe() {
        starts.push(executable);
    }
    for start in starts {
        for parent in start.ancestors() {
            if parent.join("editor/bridge.py").is_file() && parent.join("src/robbie").is_dir() {
                return Ok(parent.to_path_buf());
            }
        }
    }
    Err("Could not find Project Robbie. Set ROBBIE_PROJECT_DIR to the checkout.".into())
}

fn python_executable(root: &Path) -> Result<PathBuf, String> {
    if let Some(python) = std::env::var_os("ROBBIE_PYTHON") {
        return Ok(PathBuf::from(python));
    }
    for relative in [".venv/Scripts/python.exe", ".venv/bin/python"] {
        let path = root.join(relative);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err("The Robbie Python environment is missing. Create .venv or set ROBBIE_PYTHON.".into())
}

struct BridgeWorker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    sequence: u64,
}

impl BridgeWorker {
    fn start() -> Result<Self, String> {
        let root = project_root()?;
        let python = python_executable(&root)?;
        let mut command = Command::new(&python);
        command
            .arg("-u")
            .arg(root.join("editor/bridge.py"))
            .arg("--serve")
            .current_dir(&root)
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let mut child = command.spawn().map_err(|error| {
            format!("Could not start Robbie with {}: {error}", python.display())
        })?;
        let input = child.stdin.take().ok_or("Could not open Robbie input")?;
        let output = BufReader::new(child.stdout.take().ok_or("Could not open Robbie output")?);
        Ok(Self {
            child,
            input,
            output,
            sequence: 0,
        })
    }

    fn request(
        &mut self,
        command: &str,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        self.sequence += 1;
        let request =
            serde_json::json!({"id": self.sequence, "command": command, "payload": payload});
        writeln!(self.input, "{request}")
            .and_then(|_| self.input.flush())
            .map_err(|error| format!("Lost connection to Robbie: {error}"))?;
        let mut response = String::new();
        let size = self
            .output
            .read_line(&mut response)
            .map_err(|error| format!("Could not read Robbie's response: {error}"))?;
        if size == 0 {
            return Err(
                "Robbie's Python process stopped. Try the request again to reconnect.".into(),
            );
        }
        let envelope: serde_json::Value = serde_json::from_str(&response)
            .map_err(|error| format!("Invalid response from Robbie: {error}"))?;
        if envelope["id"].as_u64() != Some(self.sequence) {
            return Err("Robbie response did not match the request.".into());
        }
        envelope
            .get("result")
            .cloned()
            .ok_or("Robbie returned no result".into())
    }
}

impl Drop for BridgeWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn invoke(command: &str, payload_json: &str) -> Result<String, String> {
    let payload =
        serde_json::from_str(payload_json).map_err(|error| format!("Invalid payload: {error}"))?;
    let mut worker = WORKER
        .lock()
        .map_err(|_| "Robbie's connection lock is unavailable")?;
    if worker.is_none() {
        *worker = Some(BridgeWorker::start()?);
    }
    let response = worker
        .as_mut()
        .ok_or("Robbie is unavailable")?
        .request(command, payload);
    let result = match response {
        Ok(result) => result,
        Err(error) => {
            // Reconnect on the next request, never silently replay an action.
            *worker = None;
            return Err(error);
        }
    };
    if let Some(error) = result.get("error").and_then(|value| value.as_str()) {
        return Err(error.to_owned());
    }
    Ok(result.to_string())
}

#[cfg(test)]
mod tests {
    use super::invoke;

    #[test]
    fn native_bridge_reuses_python_model_and_session() {
        let status: serde_json::Value =
            serde_json::from_str(&invoke("status", "{}").unwrap()).unwrap();
        assert_eq!(status["loaded"], true);
        assert!(status["parameters"].as_u64().unwrap() > 0);
        invoke("reset", "{}").unwrap();
        let first: serde_json::Value =
            serde_json::from_str(&invoke("chat", r#"{"message":"whats the time now?"}"#).unwrap())
                .unwrap();
        assert_eq!(first["turns"], 2);
        assert!(first["reply"].as_str().unwrap().contains("UTC"));
        let second: serde_json::Value =
            serde_json::from_str(&invoke("chat", r#"{"message":"What is RAM?"}"#).unwrap())
                .unwrap();
        assert_eq!(second["turns"], 4);
        assert_eq!(second["checkpoint"], status["checkpoint"]);
        assert!(!second["reply"].as_str().unwrap().is_empty());
        assert!(invoke("no_such_command", "{}").is_err());
        let reset: serde_json::Value =
            serde_json::from_str(&invoke("reset", "{}").unwrap()).unwrap();
        assert_eq!(reset["turns"], 0);
    }
}
