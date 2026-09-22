//! Isolated access to the production concept factory and extraction math.
//! Deliberately cannot generate previews or publish a vector into the mixer.
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use torment_nexus::{
    artifacts::{ArtifactStore, sha256},
    codex::CodexClient,
    factory, steering,
};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Canonicalize {
        #[arg(long)]
        factory: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Factory {
        #[arg(long)]
        concept: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "codex")]
        codex: PathBuf,
    },
    Analyze {
        #[arg(long)]
        paper: bool,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        captures: PathBuf,
        #[arg(long)]
        fingerprint: String,
        #[arg(long)]
        output: PathBuf,
    },
}

fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(
        &fs::read(path).with_context(|| path.display().to_string())?,
    )?)
}

fn write_atomic(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
    let mut file = File::create(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    File::open(path.parent().context("output parent")?)?.sync_all()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    match Args::parse().command {
        Command::Canonicalize { factory, output } => {
            fs::create_dir_all(&output)?;
            let original = read(&factory)?;
            let (pairs, aliases) = factory::canonicalize_families(
                original["dataset"].as_array().context("dataset")?,
                &original["design"],
            )?;
            write_atomic(
                &output.join("prepared.json"),
                &steering::prepare_dataset(&json!(pairs))?,
            )?;
            write_atomic(
                &output.join("family-correction.json"),
                &json!({"source_factory_sha256":sha256(&fs::read(factory)?),
                "algorithm":"designer-exact-or-unique-short-name-v1","aliases":aliases,"text_examples_unchanged":true}),
            )?;
        }
        Command::Factory {
            concept,
            output,
            codex,
        } => {
            fs::create_dir_all(&output)?;
            let output = output.canonicalize()?;
            // An OS-held lock prevents two resumptions purchasing the same stage.
            let lock = File::create(output.join("factory.lock"))?;
            lock.try_lock()
                .context("another factory owns this benchmark directory")?;
            let text = fs::read_to_string(concept)?;
            let binding = output.join("concept.json");
            let identity = json!({"text":text,"sha256":sha256(text.as_bytes())});
            if binding.exists() {
                ensure!(
                    read(&binding)? == identity,
                    "concept changed; use a new output directory"
                );
            } else {
                write_atomic(&binding, &identity)?;
            }
            let artifacts = ArtifactStore::open(output.join("artifacts"))?;
            let index_path = output.join("stages.json");
            let index: BTreeMap<String, String> = if index_path.exists() {
                serde_json::from_value(read(&index_path)?)?
            } else {
                BTreeMap::new()
            };
            let stages = index
                .iter()
                .map(|(stage, hash)| Ok((stage.clone(), artifacts.read_json(hash)?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            if output.join("factory.json").exists() {
                println!("Completed dataset retained; no cloud calls made.");
                return Ok(());
            }
            let index = Mutex::new(index);
            let cancelled = Arc::new(AtomicBool::new(false));
            let flag = cancelled.clone();
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                flag.store(true, Ordering::Release);
            });
            let (send, mut receive) = tokio::sync::mpsc::unbounded_channel::<Value>();
            let progress = tokio::spawn(async move {
                while let Some(event) = receive.recv().await {
                    if event["status"].is_string() {
                        eprintln!("{} {}", event["stage"], event["status"]);
                    }
                }
            });
            let client = CodexClient::new(output.join("codex")).with_executable(codex);
            let result = factory::run(
                &client,
                &text,
                None,
                stages,
                cancelled,
                send,
                |stage, value| {
                    let hash = artifacts.put_json(value)?;
                    let mut index = index
                        .lock()
                        .map_err(|_| anyhow::anyhow!("checkpoint lock poisoned"))?;
                    index.insert(stage.to_owned(), hash);
                    write_atomic(&index_path, &serde_json::to_value(&*index)?)
                },
            )
            .await;
            progress.abort();
            let result = result?;
            write_atomic(
                &output.join("prepared.json"),
                &steering::prepare_dataset(&result.dataset)?,
            )?;
            write_atomic(
                &output.join("factory.json"),
                &serde_json::to_value(&result)?,
            )?;
            println!(
                "Completed {} pairs; no inference, previews, or publication.",
                result.dataset.as_array().context("dataset")?.len()
            );
        }
        Command::Analyze {
            paper,
            dataset,
            captures,
            fingerprint,
            output,
        } => {
            let dataset = read(&dataset)?;
            let pairs = dataset.get("pairs").unwrap_or(&dataset);
            let analysis =
                steering::analyze_with_method(pairs, &read(&captures)?, &fingerprint, paper)?;
            write_atomic(&output, &analysis)?;
        }
    }
    Ok(())
}
