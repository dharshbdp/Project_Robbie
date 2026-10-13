use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Locate the Python interpreter associated with Project Robbie (.venv).
fn find_python_executable() -> PathBuf {
    let candidates = [
        "../../.venv/Scripts/python.exe",
        "../.venv/Scripts/python.exe",
        ".venv/Scripts/python.exe",
        "../../.venv/bin/python",
        "../.venv/bin/python",
    ];

    for c in &candidates {
        let p = Path::new(c);
        if p.exists() {
            if let Ok(canon) = p.canonicalize() {
                return canon;
            }
            return p.to_path_buf();
        }
    }

    PathBuf::from("python")
}

/// Locate the bridge script path.
fn find_bridge_script() -> PathBuf {
    let candidates = [
        "../bridge.py",
        "editor/bridge.py",
        "./bridge.py",
    ];

    for c in &candidates {
        let p = Path::new(c);
        if p.exists() {
            if let Ok(canon) = p.canonicalize() {
                return canon;
            }
            return p.to_path_buf();
        }
    }

    PathBuf::from("editor/bridge.py")
}

/// Execute a bridge command with JSON payload piped to stdin.
fn invoke_bridge(subcmd: &str, payload_json: &str) -> Result<String, String> {
    let python = find_python_executable();
    let bridge = find_bridge_script();

    let mut child = Command::new(&python)
        .arg(&bridge)
        .arg(subcmd)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn Python process {:?}: {}", python, e))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(payload_json.as_bytes())
            .map_err(|e| format!("Failed to write to Python stdin: {}", e))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to read Python process output: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Bridge execution failed (code {:?}): {}", output.status.code(), stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.trim().to_string())
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Robbie Studio is online and connected.", name)
}

#[tauri::command]
fn ask_robbie(prompt: String) -> Result<String, String> {
    let payload = serde_json::json!({ "message": prompt }).to_string();
    invoke_bridge("chat", &payload)
}

#[tauri::command]
fn get_system_telemetry() -> Result<String, String> {
    invoke_bridge("system", "{}")
}

#[tauri::command]
fn get_memory_profile() -> Result<String, String> {
    invoke_bridge("memory", "{}")
}

#[tauri::command]
fn manage_memory(
    action: Option<String>,
    key: Option<String>,
    value: Option<String>,
    title: Option<String>,
    content: Option<String>,
) -> Result<String, String> {
    let payload = serde_json::json!({
        "action": action.unwrap_or_else(|| "get".to_string()),
        "key": key,
        "value": value,
        "title": title,
        "content": content
    }).to_string();
    invoke_bridge("memory", &payload)
}

#[tauri::command]
fn run_research(topic: String) -> Result<String, String> {
    let payload = serde_json::json!({ "topic": topic }).to_string();
    invoke_bridge("research", &payload)
}

#[tauri::command]
fn execute_code(code: String) -> Result<String, String> {
    let payload = serde_json::json!({ "code": code }).to_string();
    invoke_bridge("execute", &payload)
}

#[tauri::command]
fn launch_app(app: String) -> Result<String, String> {
    let payload = serde_json::json!({ "app": app }).to_string();
    invoke_bridge("launch", &payload)
}

#[tauri::command]
#[allow(non_snake_case)]
fn run_diagnostic(
    diag_type: Option<String>,
    diagType: Option<String>,
    query: Option<String>,
) -> Result<String, String> {
    let target = query
        .or(diagType)
        .or(diag_type)
        .unwrap_or_else(|| "all".to_string());
    let payload = serde_json::json!({
        "query": target,
        "type": target
    }).to_string();
    invoke_bridge("diagnose", &payload)
}

#[tauri::command]
fn browse_files(action: String, path: String, pattern: Option<String>) -> Result<String, String> {
    let payload = serde_json::json!({
        "action": action,
        "path": path,
        "pattern": pattern.unwrap_or_else(|| "*".to_string())
    }).to_string();
    invoke_bridge("files", &payload)
}

#[tauri::command]
fn read_project_file(file_path: String) -> Result<String, String> {
    std::fs::read_to_string(&file_path).map_err(|e| format!("Could not read file {}: {}", file_path, e))
}

#[tauri::command]
fn write_project_file(_file_path: String, _content: String) -> Result<bool, String> {
    Err("Safety Policy Enforced: Robbie operates under strict read-only computer safety. Modifying or writing files is disabled.".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            ask_robbie,
            get_system_telemetry,
            get_memory_profile,
            manage_memory,
            run_research,
            execute_code,
            launch_app,
            run_diagnostic,
            browse_files,
            read_project_file,
            write_project_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
