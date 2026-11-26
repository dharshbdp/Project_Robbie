mod bridge;

async fn invoke_bridge(command: &str, payload: &str) -> Result<String, String> {
    let command = command.to_owned();
    let payload = payload.to_owned();
    tauri::async_runtime::spawn_blocking(move || bridge::invoke(&command, &payload))
        .await
        .map_err(|error| format!("Robbie connection task failed: {error}"))?
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Robbie Studio is online and connected.", name)
}

#[tauri::command]
async fn ask_robbie(prompt: String) -> Result<String, String> {
    let payload = serde_json::json!({ "message": prompt }).to_string();
    invoke_bridge("chat", &payload).await
}

#[tauri::command]
async fn get_system_telemetry() -> Result<String, String> {
    invoke_bridge("system", "{}").await
}

#[tauri::command]
async fn get_memory_profile() -> Result<String, String> {
    invoke_bridge("memory", "{}").await
}

#[tauri::command]
async fn get_backend_status() -> Result<String, String> {
    invoke_bridge("status", "{}").await
}

#[tauri::command]
async fn reset_chat() -> Result<String, String> {
    invoke_bridge("reset", "{}").await
}

#[tauri::command]
async fn manage_memory(
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
    })
    .to_string();
    invoke_bridge("memory", &payload).await
}

#[tauri::command]
async fn run_research(topic: String) -> Result<String, String> {
    let payload = serde_json::json!({ "topic": topic }).to_string();
    invoke_bridge("research", &payload).await
}

#[tauri::command]
async fn execute_code(code: String) -> Result<String, String> {
    let payload = serde_json::json!({ "code": code }).to_string();
    invoke_bridge("execute", &payload).await
}

#[tauri::command]
async fn launch_app(app: String) -> Result<String, String> {
    let payload = serde_json::json!({ "app": app }).to_string();
    invoke_bridge("launch", &payload).await
}

#[tauri::command]
#[allow(non_snake_case)]
async fn run_diagnostic(
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
    })
    .to_string();
    invoke_bridge("diagnose", &payload).await
}

#[tauri::command]
async fn browse_files(
    action: String,
    path: String,
    pattern: Option<String>,
) -> Result<String, String> {
    let payload = serde_json::json!({
        "action": action,
        "path": path,
        "pattern": pattern.unwrap_or_else(|| "*".to_string())
    })
    .to_string();
    invoke_bridge("files", &payload).await
}

#[tauri::command]
async fn read_project_file(file_path: String) -> Result<String, String> {
    let payload = serde_json::json!({"action": "read", "path": file_path}).to_string();
    let raw = invoke_bridge("files", &payload).await?;
    let result: serde_json::Value =
        serde_json::from_str(&raw).map_err(|error| error.to_string())?;
    result["content"]
        .as_str()
        .map(str::to_owned)
        .ok_or("The file could not be read".into())
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
            get_backend_status,
            reset_chat,
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
