//! Versioned private-pipe transport. The C++ child alone owns inference state.
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    path::Path,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{Mutex, mpsc},
};
use uuid::Uuid;

type Waiters = Arc<Mutex<HashMap<String, mpsc::UnboundedSender<Value>>>>;

#[derive(Clone)]
pub struct Engine {
    input: Arc<Mutex<ChildStdin>>,
    waiters: Waiters,
    alive: Arc<AtomicBool>,
    child: Arc<Mutex<Child>>,
}

impl Engine {
    pub async fn start(binary: &Path, logs: &Path) -> Result<Self> {
        ensure!(
            binary.is_file(),
            "inference worker missing at {}; run scripts/build-engine.sh or use the packaged launcher",
            binary.display()
        );
        std::fs::create_dir_all(logs)?;
        let log_path = logs.join(format!("engine-{}.log", Uuid::new_v4()));
        let log = std::fs::File::create(&log_path)?;
        let mut child = Command::new(binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log))
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("start inference worker {}", binary.display()))?;
        let input = child.stdin.take().context("worker stdin missing")?;
        let output = child.stdout.take().context("worker stdout missing")?;
        let waiters: Waiters = Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let (pending, running) = (waiters.clone(), alive.clone());
        tokio::spawn(async move {
            let mut lines = BufReader::new(output).lines();
            let failure = loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        if line.len() > 64 * 1024 * 1024 {
                            break "worker event exceeds 64 MiB".to_string();
                        }
                        let value = match serde_json::from_str::<Value>(&line) {
                            Ok(v) if v["v"] == 1 && v["id"].is_string() => v,
                            _ => break "invalid worker protocol event".to_string(),
                        };
                        let id = value["id"].as_str().unwrap();
                        let terminal =
                            matches!(value["event"].as_str(), Some("result" | "done" | "error"));
                        let mut map = pending.lock().await;
                        if let Some(sender) = map.get(id) {
                            let _ = sender.send(value.clone());
                        }
                        if terminal {
                            map.remove(id);
                        }
                    }
                    Ok(None) => break "worker exited; reload the model to restart it".to_string(),
                    Err(e) => break format!("worker pipe failed: {e}"),
                }
            };
            running.store(false, Ordering::Release);
            let mut map = pending.lock().await;
            for (id, sender) in map.drain() {
                let _ = sender.send(json!({"v":1,"id":id,"event":"error","error":format!("{failure}; log: {}",log_path.display())}));
            }
        });
        Ok(Self {
            input: Arc::new(Mutex::new(input)),
            waiters,
            alive,
            child: Arc::new(Mutex::new(child)),
        })
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    pub async fn send(
        &self,
        mut command: Value,
    ) -> Result<(String, mpsc::UnboundedReceiver<Value>)> {
        ensure!(self.is_alive(), "inference worker exited; reload the model");
        let id = command["id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        command["id"] = json!(id);
        command["v"] = json!(1);
        let mut encoded = serde_json::to_vec(&command)?;
        ensure!(
            encoded.len() <= 64 * 1024 * 1024,
            "worker command exceeds 64 MiB"
        );
        encoded.push(b'\n');
        let (sender, receiver) = mpsc::unbounded_channel();
        {
            let mut waiters = self.waiters.lock().await;
            ensure!(!waiters.contains_key(&id), "duplicate engine command id");
            waiters.insert(id.clone(), sender);
        }
        let write = async {
            let mut input = self.input.lock().await;
            input.write_all(&encoded).await?;
            input.flush().await
        }
        .await;
        if let Err(error) = write {
            self.waiters.lock().await.remove(&id);
            return Err(error).context("write worker command");
        }
        Ok((id, receiver))
    }

    pub async fn execute(&self, command: Value) -> Result<Value> {
        let (_, events) = self.send(command).await?;
        Self::result(events).await
    }

    /// The job publishes its target before calling this method. A cancellation
    /// flag set before enqueue must also cancel the now-enqueued command, not
    /// merely an earlier idle worker state. Later cancellation reads that target.
    pub async fn send_cancellable(
        &self,
        command: Value,
        cancelled: &AtomicBool,
    ) -> Result<(String, mpsc::UnboundedReceiver<Value>)> {
        let (id, events) = self.send(command).await?;
        if cancelled.load(Ordering::Acquire) {
            self.cancel(&id).await?;
        }
        Ok((id, events))
    }

    pub async fn execute_cancellable(
        &self,
        command: Value,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        let (_, events) = self.send_cancellable(command, cancelled).await?;
        Self::result(events).await
    }

    async fn result(mut events: mpsc::UnboundedReceiver<Value>) -> Result<Value> {
        while let Some(event) = events.recv().await {
            match event["event"].as_str() {
                Some("result" | "done") => return Ok(event),
                Some("error") => bail!("{}", event["error"].as_str().unwrap_or("worker error")),
                _ => {}
            }
        }
        bail!("worker event channel closed without a result")
    }

    pub async fn cancel(&self, target: &str) -> Result<()> {
        self.execute(json!({"op":"cancel","target":target})).await?;
        Ok(())
    }

    pub async fn shutdown(&self) {
        let _ = self.child.lock().await.kill().await;
        self.alive.store(false, Ordering::Release);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{os::unix::fs::PermissionsExt, time::Duration};
    use tokio::time::timeout;

    const DEADLINE: Duration = Duration::from_secs(10);

    async fn fixture() -> (tempfile::TempDir, Engine) {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("engine-worker");
        std::fs::write(
            &executable,
            include_str!("../tests/fixtures/engine_worker.py"),
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let engine = Engine::start(&executable, &directory.path().join("logs"))
            .await
            .unwrap();
        (directory, engine)
    }

    async fn registered_while_enqueue_is_blocked(engine: &Engine, id: &str) {
        // The caller holds input's mutex. Seeing the waiter proves send() has
        // reached its enqueue boundary but cannot have written the command.
        // This handshake, not a sleep, establishes the race ordering.
        timeout(DEADLINE, async {
            loop {
                if engine.waiters.lock().await.contains_key(id) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("send never reached the blocked enqueue boundary");
    }

    async fn drain_closed(mut events: mpsc::UnboundedReceiver<Value>) -> Vec<Value> {
        timeout(DEADLINE, async {
            let mut received = Vec::new();
            while let Some(event) = events.recv().await {
                received.push(event);
            }
            received
        })
        .await
        .expect("terminal event did not close the command channel")
    }

    fn assert_job_then_cancel(terminal: &Value, op: &str, job_id: &str) {
        let observed = terminal["observed"].as_array().unwrap();
        assert_eq!(observed.len(), 2, "{terminal}");
        assert_eq!(observed[0]["op"], op);
        assert_eq!(observed[0]["id"], job_id);
        assert_eq!(observed[1]["op"], "cancel");
        assert_eq!(observed[1]["target"], job_id);
        assert_eq!(terminal["cancelled"], true);
    }

    #[tokio::test]
    async fn send_cancellable_cancels_after_blocked_enqueue_and_closes_events() {
        let (_directory, engine) = fixture().await;
        let input = engine.input.lock().await;
        let cancelled = Arc::new(AtomicBool::new(false));
        let sender = engine.clone();
        let flag = cancelled.clone();
        let task = tokio::spawn(async move {
            sender
                .send_cancellable(
                    json!({"id":"generation","op":"generate","terminal_order":"ack_then_done"}),
                    &flag,
                )
                .await
        });
        registered_while_enqueue_is_blocked(&engine, "generation").await;
        assert!(!task.is_finished());
        cancelled.store(true, Ordering::Release);
        drop(input);

        let (id, events) = timeout(DEADLINE, task).await.unwrap().unwrap().unwrap();
        assert_eq!(id, "generation");
        let events = drain_closed(events).await;
        assert_eq!(events.first().unwrap()["event"], "progress");
        assert_eq!(events.last().unwrap()["event"], "done");
        assert_job_then_cancel(events.last().unwrap(), "generate", "generation");
        assert!(engine.waiters.lock().await.is_empty());
        assert!(
            engine.is_alive(),
            "channel closure must not depend on worker exit"
        );
        engine.shutdown().await;
    }

    #[tokio::test]
    async fn execute_cancellable_retains_terminal_arriving_before_cancel_ack() {
        let (_directory, engine) = fixture().await;
        let input = engine.input.lock().await;
        let cancelled = Arc::new(AtomicBool::new(false));
        let sender = engine.clone();
        let flag = cancelled.clone();
        let task = tokio::spawn(async move {
            sender
                .execute_cancellable(
                    json!({"id":"extraction","op":"extract","terminal_order":"done_then_ack"}),
                    &flag,
                )
                .await
        });
        registered_while_enqueue_is_blocked(&engine, "extraction").await;
        assert!(!task.is_finished());
        cancelled.store(true, Ordering::Release);
        drop(input);

        let terminal = timeout(DEADLINE, task).await.unwrap().unwrap().unwrap();
        assert_eq!(terminal["event"], "done");
        assert_job_then_cancel(&terminal, "extract", "extraction");
        assert!(engine.waiters.lock().await.is_empty());
        let inspected = engine.execute(json!({"op":"inspect"})).await.unwrap();
        assert!(inspected["active"].is_null());
        assert_eq!(inspected["observed"].as_array().unwrap().len(), 3);
        assert!(engine.waiters.lock().await.is_empty());
        engine.shutdown().await;
    }

    #[tokio::test]
    async fn uncancelled_result_and_error_leave_no_waiters_or_extra_cancel() {
        let (_directory, engine) = fixture().await;
        let cancelled = AtomicBool::new(false);
        let (_, events) = engine
            .send_cancellable(json!({"id":"normal","op":"complete"}), &cancelled)
            .await
            .unwrap();
        let events = drain_closed(events).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["event"], "result");
        assert_eq!(events[0]["observed"].as_array().unwrap().len(), 1);
        assert!(engine.waiters.lock().await.is_empty());

        let error = engine
            .execute_cancellable(json!({"id":"failure","op":"fail"}), &cancelled)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("intentional fixture failure"));
        assert!(engine.waiters.lock().await.is_empty());
        let inspected = engine.execute(json!({"op":"inspect"})).await.unwrap();
        let observed = inspected["observed"].as_array().unwrap();
        assert_eq!(observed.len(), 3);
        assert!(observed.iter().all(|command| command["op"] != "cancel"));
        assert!(inspected["active"].is_null());
        assert!(engine.waiters.lock().await.is_empty());
        engine.shutdown().await;
    }
}
