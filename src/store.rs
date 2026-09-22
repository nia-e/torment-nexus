//! Durable record index. Large payloads belong in the artifact store; immutable
//! recipe/vector versions are inserted once, while jobs/runs are snapshots.

use crate::artifacts::ArtifactStore;
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const RECORD_KINDS: &[&str] = &[
    "models",
    "recipes",
    "vectors",
    "vector_visibility",
    "record_metadata",
    "presets",
    "jobs",
    "runs",
    "conversations",
];

#[derive(Clone)]
pub struct Store {
    root: Arc<PathBuf>,
    connection: Arc<Mutex<Connection>>,
    artifacts: ArtifactStore,
    // Drop after the connection, so opening the next instance cannot overlap
    // SQLite's last-connection cleanup.
    _instance_lock: Arc<std::fs::File>,
}

impl Store {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let mut directories = std::fs::DirBuilder::new();
        directories.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            directories.mode(0o700);
        }
        directories
            .create(root.as_ref())
            .context("create application data directory")?;
        let root = root.as_ref().canonicalize()?;
        let instance_lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("instance.lock"))?;
        instance_lock
            .try_lock()
            .context("this data directory is already open in another Torment Nexus process")?;
        let mut connection =
            Connection::open(root.join("index.sqlite3")).context("open application database")?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.pragma_update(None, "foreign_keys", true)?;
        let version: i64 = connection.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            version <= 1,
            "database version {version} is newer than this application supports"
        );
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL,
                id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (kind, id)
             );
             CREATE INDEX IF NOT EXISTS records_by_kind_created ON records(kind, created_at);
             PRAGMA user_version = 1;",
        )?;
        mark_interrupted(&mut connection)?;
        let artifacts = ArtifactStore::open(root.join("artifacts"))?;
        Ok(Self {
            root: Arc::new(root),
            _instance_lock: Arc::new(instance_lock),
            connection: Arc::new(Mutex::new(connection)),
            artifacts,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn artifacts(&self) -> &ArtifactStore {
        &self.artifacts
    }

    fn connection(&self) -> Result<MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex was poisoned"))
    }

    pub fn put(&self, kind: &str, id: &str, value: &Value) -> Result<()> {
        self.put_records(&[(kind, id, value)])
    }

    /// Commit related snapshots together. Any validation or immutability failure
    /// rolls the whole batch back; artifacts must already be durably installed.
    pub fn put_batch(&self, records: &[(&str, &str, Value)]) -> Result<()> {
        let references: Vec<_> = records
            .iter()
            .map(|(kind, id, value)| (*kind, *id, value))
            .collect();
        self.put_records(&references)
    }

    fn put_records(&self, records: &[(&str, &str, &Value)]) -> Result<()> {
        let mut payloads = Vec::with_capacity(records.len());
        for (kind, id, value) in records {
            validate_kind(kind)?;
            ensure!(
                !id.is_empty() && id.len() <= 256,
                "record id is empty or too long"
            );
            ensure!(value.is_object(), "record must be a JSON object");
            ensure!(
                value.get("id").and_then(Value::as_str) == Some(*id),
                "record id does not match its index key"
            );
            validate_artifact_references(&self.artifacts, value)?;
            let payload = serde_json::to_string(value)?;
            ensure!(
                payload.len() <= 32 * 1024 * 1024,
                "record exceeds 32 MiB; use an artifact"
            );
            payloads.push(payload);
        }
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        for ((kind, id, value), payload) in records.iter().zip(&payloads) {
            if matches!(*kind, "recipes" | "vectors") {
                let existing: Option<String> = transaction
                    .query_row(
                        "SELECT payload FROM records WHERE kind = ?1 AND id = ?2",
                        params![kind, id],
                        |row| row.get(0),
                    )
                    .optional()?;
                if let Some(existing) = existing {
                    ensure!(
                        serde_json::from_str::<Value>(&existing)? == **value,
                        "{kind} versions are immutable; create a new version id"
                    );
                    continue;
                }
            }
            let now = now_seconds();
            let created_at = value
                .get("created_at")
                .and_then(Value::as_i64)
                .unwrap_or(now);
            transaction.execute(
                "INSERT INTO records (kind, id, payload, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(kind, id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at",
                params![kind, id, payload, created_at, now],
            )?;
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn get(&self, kind: &str, id: &str) -> Result<Option<Value>> {
        validate_kind(kind)?;
        let connection = self.connection()?;
        let result: Option<String> = connection
            .query_row(
                "SELECT payload FROM records WHERE kind = ?1 AND id = ?2",
                params![kind, id],
                |row| row.get(0),
            )
            .optional()?;
        result
            .map(|raw| serde_json::from_str(&raw).context("invalid stored record JSON"))
            .transpose()
    }

    pub fn list(&self, kind: &str) -> Result<Vec<Value>> {
        validate_kind(kind)?;
        let connection = self.connection()?;
        list_records(&connection, kind)
    }

    /// A consistent database snapshot. Live engine/Codex state is added by the app.
    pub fn snapshot(&self) -> Result<Value> {
        let connection = self.connection()?;
        let mut snapshot = serde_json::Map::new();
        for kind in RECORD_KINDS {
            snapshot.insert(
                (*kind).to_string(),
                Value::Array(list_records(&connection, kind)?),
            );
        }
        Ok(Value::Object(snapshot))
    }
}

pub fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn validate_kind(kind: &str) -> Result<()> {
    ensure!(RECORD_KINDS.contains(&kind), "unknown record kind: {kind}");
    Ok(())
}

fn list_records(connection: &Connection, kind: &str) -> Result<Vec<Value>> {
    let mut statement = connection
        .prepare("SELECT payload FROM records WHERE kind = ?1 ORDER BY created_at DESC, id ASC")?;
    let rows = statement.query_map([kind], |row| row.get::<_, String>(0))?;
    rows.map(|row| serde_json::from_str(&row?).context("invalid stored record JSON"))
        .collect()
}

fn mark_interrupted(connection: &mut Connection) -> Result<()> {
    let transaction = connection.transaction()?;
    let mut updates = Vec::new();
    for kind in ["jobs", "runs"] {
        for mut value in list_records(&transaction, kind)? {
            if matches!(
                value.get("status").and_then(Value::as_str),
                Some("running" | "queued" | "streaming")
            ) {
                let id = value
                    .get("id")
                    .and_then(Value::as_str)
                    .context("stored record has no id")?
                    .to_owned();
                value["status"] = json!("interrupted");
                value["interrupted_at"] = json!(now_seconds());
                if value.get("error").is_none_or(Value::is_null) {
                    value["error"] = json!(
                        "Application stopped before this work completed. Completed stages and partial output were retained; retry explicitly."
                    );
                }
                updates.push((kind, id, serde_json::to_string(&value)?));
            }
        }
    }
    for (kind, id, payload) in updates {
        transaction.execute(
            "UPDATE records SET payload = ?1, updated_at = ?2 WHERE kind = ?3 AND id = ?4",
            params![payload, now_seconds(), kind, id],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

fn validate_artifact_references(artifacts: &ArtifactStore, value: &Value) -> Result<()> {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if matches!(
                    key.as_str(),
                    "manifest_hash" | "direction_hash" | "unit_hash" | "artifact_hash"
                ) && !value.is_null()
                {
                    let hash = value
                        .as_str()
                        .context("artifact reference must be a hash string")?;
                    artifacts.read_bytes(hash).with_context(|| {
                        format!(
                            "{key} must reference a complete checked artifact before committing"
                        )
                    })?;
                } else {
                    validate_artifact_references(artifacts, value)?;
                }
            }
        }
        Value::Array(array) => {
            for item in array {
                validate_artifact_references(artifacts, item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reopen_preserves_outputs_and_marks_only_unfinished_work() -> Result<()> {
        let root = tempfile::tempdir()?;
        {
            let store = Store::open(root.path())?;
            store.put("jobs", "j", &json!({"id":"j","status":"running","stage":"writer-2","details":{"writer-1":"done"}}))?;
            store.put("runs", "r", &json!({"id":"r","status":"running","output":"partial output","requested_controls":[{"revision":2}]}))?;
            store.put("jobs", "done", &json!({"id":"done","status":"completed"}))?;
        }
        let store = Store::open(root.path())?;
        let job = store.get("jobs", "j")?.unwrap();
        assert_eq!(job["status"], "interrupted");
        assert_eq!(job["details"]["writer-1"], "done");
        let run = store.get("runs", "r")?.unwrap();
        assert_eq!(run["output"], "partial output");
        assert_eq!(run["requested_controls"][0]["revision"], 2);
        assert_eq!(store.get("jobs", "done")?.unwrap()["status"], "completed");
        assert_eq!(store.snapshot()?["runs"].as_array().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn immutable_versions_and_atomic_artifact_references() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = Store::open(root.path())?;
        let first = json!({"id":"v1","name":"same concept","version":1});
        store.put("recipes", "v1", &first)?;
        store.put("recipes", "v1", &first)?;
        assert!(
            store
                .put("recipes", "v1", &json!({"id":"v1","version":2}))
                .is_err()
        );
        store.put(
            "recipes",
            "v2",
            &json!({"id":"v2","name":"same concept","version":2,"parent_id":"v1"}),
        )?;
        assert_eq!(store.list("recipes")?.len(), 2);
        let missing = json!({"id":"vector","manifest_hash":"0".repeat(64)});
        assert!(store.put("vectors", "vector", &missing).is_err());
        assert!(store.get("vectors", "vector")?.is_none());
        let hash = store.artifacts().put_json(&json!({"complete":true}))?;
        store.put(
            "vectors",
            "vector",
            &json!({"id":"vector","manifest_hash":hash}),
        )?;
        assert!(store.put("jobs", "j", &json!({"id":"wrong"})).is_err());
        assert!(Store::open(root.path()).is_err());
        Ok(())
    }

    #[test]
    fn vector_visibility_is_mutable_and_survives_reopen_without_changing_versions() -> Result<()> {
        let root = tempfile::tempdir()?;
        let vector = json!({"id":"vector","name":"Test concept","version":1});
        let preset = json!({"id":"preset","axes":[{"vector_id":"vector","percent":0}]});
        let run =
            json!({"id":"run","status":"completed","axes":[{"vector_id":"vector","percent":0}]});
        {
            let store = Store::open(root.path())?;
            store.put("vectors", "vector", &vector)?;
            store.put("presets", "preset", &preset)?;
            store.put("runs", "run", &run)?;
            store.put(
                "vector_visibility",
                "vector",
                &json!({"id":"vector","archived":true}),
            )?;
        }
        let store = Store::open(root.path())?;
        assert_eq!(
            store.snapshot()?["vector_visibility"],
            json!([{"id":"vector","archived":true}])
        );
        assert_eq!(store.snapshot()?["vectors"], json!([vector.clone()]));
        store.put(
            "vector_visibility",
            "vector",
            &json!({"id":"vector","archived":false}),
        )?;
        assert_eq!(
            store.get("vector_visibility", "vector")?.unwrap()["archived"],
            false
        );
        assert_eq!(store.get("vectors", "vector")?.unwrap(), vector);
        assert_eq!(store.get("presets", "preset")?.unwrap(), preset);
        assert_eq!(store.get("runs", "run")?.unwrap(), run);
        assert!(
            store
                .put("vectors", "vector", &json!({"id":"vector","archived":true}))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn batch_rolls_back_all_rows_on_late_immutable_conflict() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = Store::open(root.path())?;
        let recipe = json!({"id":"recipe","version":1});
        store.put("recipes", "recipe", &recipe)?;
        store.put("runs", "run", &json!({"id":"run","status":"running"}))?;
        assert!(
            store
                .put_batch(&[
                    ("runs", "run", json!({"id":"run","status":"completed"})),
                    ("jobs", "new", json!({"id":"new","status":"completed"})),
                    ("recipes", "recipe", json!({"id":"recipe","version":2})),
                ])
                .is_err()
        );
        assert_eq!(store.get("runs", "run")?.unwrap()["status"], "running");
        assert!(store.get("jobs", "new")?.is_none());
        assert_eq!(store.get("recipes", "recipe")?.unwrap(), recipe);
        store.put_batch(&[
            ("recipes", "recipe", recipe), // Idempotent row must not skip later rows.
            ("runs", "run", json!({"id":"run","status":"completed"})),
            ("jobs", "new", json!({"id":"new","status":"completed"})),
        ])?;
        assert_eq!(store.get("runs", "run")?.unwrap()["status"], "completed");
        assert!(store.get("jobs", "new")?.is_some());
        Ok(())
    }
}
