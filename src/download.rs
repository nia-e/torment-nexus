//! Model acquisition. Resumable data is pinned to a Hub commit, never a mutable
//! branch URL. Local imports reference the original file rather than copying it.

use crate::artifacts::{sha256, validate_hash};
use crate::store::now_seconds;
use anyhow::{Context, Result, bail, ensure};
use futures_util::StreamExt;
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

const HF_ORIGIN: &str = "https://huggingface.co";
const MAX_METADATA_BYTES: usize = 16 * 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 512 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
    pub revision: String,
    pub phase: String,
}

pub type ProgressCallback = Arc<dyn Fn(DownloadProgress) + Send + Sync>;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PinnedDownload {
    version: u32,
    repo: String,
    file: String,
    requested_revision: String,
    revision: String,
    size_bytes: Option<u64>,
    sha256: Option<String>,
}

fn hf_client() -> Result<Client> {
    Client::builder()
        .user_agent("torment-nexus/0.1")
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(90))
        .https_only(true)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .context("create HTTPS model-download client")
}

pub async fn browse_hf(repo: &str) -> Result<Value> {
    browse_hf_revision(&hf_client()?, repo, "main").await
}

async fn browse_hf_revision(client: &Client, repo: &str, revision: &str) -> Result<Value> {
    validate_repo(repo)?;
    validate_revision_request(revision)?;
    let mut url = Url::parse(HF_ORIGIN)?;
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid Hub URL"))?;
        path.extend(["api", "models"]);
        path.extend(repo.split('/'));
        path.extend(["revision", revision]);
    }
    url.query_pairs_mut().append_pair("blobs", "true");
    let response = client
        .get(url)
        .send()
        .await
        .context("connect to Hugging Face")?;
    check_status(&response)?;
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read Hugging Face metadata")?;
        ensure!(
            bytes.len().saturating_add(chunk.len()) <= MAX_METADATA_BYTES,
            "Hugging Face metadata exceeds size limit"
        );
        bytes.extend_from_slice(&chunk);
    }
    parse_hf_metadata(
        repo,
        &serde_json::from_slice(&bytes).context("invalid Hugging Face metadata JSON")?,
    )
}

fn parse_hf_metadata(repo: &str, data: &Value) -> Result<Value> {
    let revision = data
        .get("sha")
        .and_then(Value::as_str)
        .context("Hugging Face did not supply an immutable revision")?;
    validate_commit(revision)?;
    let siblings = data
        .get("siblings")
        .and_then(Value::as_array)
        .context("Hugging Face did not supply a file list")?;
    let mut files = Vec::new();
    for sibling in siblings {
        let Some(name) = sibling.get("rfilename").and_then(Value::as_str) else {
            continue;
        };
        if !name.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        validate_filename(name)?;
        let size = sibling
            .get("size")
            .and_then(Value::as_u64)
            .or_else(|| sibling.pointer("/lfs/size").and_then(Value::as_u64));
        let checksum = sibling.pointer("/lfs/sha256").and_then(Value::as_str);
        if let Some(checksum) = checksum {
            validate_hash(checksum)?;
        }
        files.push(json!({"name":name,"size_bytes":size,"sha256":checksum}));
    }
    files.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Ok(json!({"repo":repo,"revision":revision,"files":files}))
}

pub fn import_model(path: &Path, name: Option<&str>) -> Result<Value> {
    let path = path
        .canonicalize()
        .with_context(|| format!("model file not found: {}", path.display()))?;
    let (fingerprint, size) = fingerprint_gguf(&path, None)?;
    let display_name = name
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
    Ok(json!({
        "id":Uuid::new_v4().to_string(),"created_at":now_seconds(),"name":display_name,
        "path":path,"fingerprint":fingerprint,"size_bytes":size,
        "source":{"kind":"local","revision":null,"path":path},
    }))
}

/// Recheck before loading an imported path; a pathname is not an immutable model.
pub fn verify_model(path: &Path, expected_fingerprint: &str) -> Result<()> {
    validate_hash(expected_fingerprint)?;
    let (actual, _) = fingerprint_gguf(path, None)?;
    ensure!(
        actual == expected_fingerprint,
        "model file changed since import; re-import it before loading or using its vectors"
    );
    Ok(())
}

pub async fn download_model(
    root: &Path,
    repo: &str,
    file: &str,
    revision: Option<&str>,
    cancel: Arc<AtomicBool>,
    progress: ProgressCallback,
) -> Result<Value> {
    validate_repo(repo)?;
    validate_filename(file)?;
    ensure!(
        file.to_ascii_lowercase().ends_with(".gguf"),
        "select a GGUF model file"
    );
    let requested_revision = revision.unwrap_or("main");
    validate_revision_request(requested_revision)?;
    ensure!(
        !cancel.load(Ordering::Relaxed),
        "download cancelled; retry retains the pinned partial file"
    );
    fs::create_dir_all(root.join("downloads"))?;
    fs::create_dir_all(root.join("models"))?;
    let root = root.canonicalize()?;
    File::open(&root)?.sync_all()?;
    let request_key = sha256(&serde_json::to_vec(&json!([
        "request",
        repo,
        file,
        requested_revision
    ]))?);
    let request_path = root.join("downloads").join(format!("{request_key}.json"));
    let _request_lock = lock_download(&root.join("downloads").join(format!("{request_key}.lock")))?;
    let client = hf_client()?;
    let pinned: PinnedDownload = if request_path.try_exists()? {
        let bytes = read_small_file(&request_path, MAX_METADATA_BYTES)?;
        let pinned: PinnedDownload =
            serde_json::from_slice(&bytes).context("resume metadata is corrupt")?;
        ensure!(
            pinned.version == 1
                && pinned.repo == repo
                && pinned.file == file
                && pinned.requested_revision == requested_revision,
            "resume metadata does not match requested model"
        );
        validate_commit(&pinned.revision)?;
        if let Some(checksum) = &pinned.sha256 {
            validate_hash(checksum)?;
        }
        pinned
    } else {
        let metadata = tokio::select! {
            _ = wait_cancelled(&cancel) => bail!("download cancelled before metadata resolution"),
            result = browse_hf_revision(&client, repo, requested_revision) => result?,
        };
        let selected = metadata["files"]
            .as_array()
            .context("Hub file list")?
            .iter()
            .find(|candidate| candidate["name"].as_str() == Some(file))
            .with_context(|| format!("GGUF file {file} was not found in repository {repo}"))?;
        let pinned = PinnedDownload {
            version: 1,
            repo: repo.into(),
            file: file.into(),
            requested_revision: requested_revision.into(),
            revision: metadata["revision"]
                .as_str()
                .context("immutable revision")?
                .into(),
            size_bytes: selected["size_bytes"].as_u64(),
            sha256: selected["sha256"].as_str().map(str::to_owned),
        };
        write_metadata_atomic(&request_path, &serde_json::to_vec(&pinned)?)?;
        pinned
    };
    if let Some(size) = pinned.size_bytes {
        ensure!(
            size <= MAX_MODEL_BYTES,
            "model exceeds 512 GiB download limit"
        );
    }
    let content_key = sha256(&serde_json::to_vec(&json!([
        "content",
        repo,
        file,
        pinned.revision
    ]))?);
    let _content_lock = lock_download(&root.join("downloads").join(format!("{content_key}.lock")))?;
    let partial = root.join("downloads").join(format!("{content_key}.part"));
    let filename = Path::new(file)
        .file_name()
        .context("model filename")?
        .to_string_lossy();
    let destination = root
        .join("models")
        .join(format!("{content_key}-{filename}"));
    let current = if destination.try_exists()? {
        fs::metadata(&destination)?.len()
    } else {
        fs::metadata(&partial).map(|m| m.len()).unwrap_or(0)
    };
    progress(DownloadProgress {
        downloaded: current,
        total: pinned.size_bytes,
        revision: pinned.revision.clone(),
        phase: "resolved".into(),
    });
    if !destination.try_exists()? {
        let mut url = Url::parse(HF_ORIGIN)?;
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| anyhow::anyhow!("invalid Hub URL"))?;
            segments.extend(repo.split('/'));
            segments.extend(["resolve", &pinned.revision]);
            segments.extend(file.split('/'));
        }
        transfer(
            &client,
            url,
            &partial,
            pinned.size_bytes,
            &cancel,
            &progress,
            &pinned.revision,
        )
        .await?;
    }
    let checking_path = if destination.try_exists()? {
        destination.clone()
    } else {
        partial.clone()
    };
    progress(DownloadProgress {
        downloaded: fs::metadata(&checking_path)?.len(),
        total: pinned.size_bytes,
        revision: pinned.revision.clone(),
        phase: "verifying".into(),
    });
    let checking_cancel = cancel.clone();
    let task_path = checking_path.clone();
    let verified =
        tokio::task::spawn_blocking(move || fingerprint_gguf(&task_path, Some(&checking_cancel)))
            .await
            .context("model verification task failed")?;
    if cancel.load(Ordering::Relaxed) {
        bail!("download cancelled during verification; completed bytes retained for retry");
    }
    let verified = verified.and_then(|(hash, size)| {
        if let Some(expected) = pinned.size_bytes {
            ensure!(
                size == expected,
                "model size differs from pinned Hub metadata"
            );
        }
        if let Some(expected) = &pinned.sha256 {
            ensure!(
                hash == *expected,
                "model SHA-256 differs from pinned Hub metadata"
            );
        }
        Ok((hash, size))
    });
    let (fingerprint, size) = match verified {
        Ok(verified) => verified,
        Err(error) if checking_path == partial => {
            let rejected = partial.with_extension(format!("rejected-{}", Uuid::new_v4()));
            fs::rename(&partial, &rejected)?;
            bail!(
                "Downloaded model failed verification: {error:#}. Suspect bytes retained at {}; retry downloads fresh.",
                rejected.display()
            );
        }
        Err(error) => {
            return Err(error
                .context("cached downloaded model failed verification; do not load this model"));
        }
    };
    if checking_path == partial {
        fs::rename(&partial, &destination).context("atomically publish verified model")?;
        File::open(destination.parent().context("model directory")?)?.sync_all()?;
    }
    fs::remove_file(&request_path)?;
    File::open(request_path.parent().context("download directory")?)?.sync_all()?;
    progress(DownloadProgress {
        downloaded: size,
        total: Some(size),
        revision: pinned.revision.clone(),
        phase: "completed".into(),
    });
    Ok(json!({
        "id":Uuid::new_v4().to_string(),"created_at":now_seconds(),"name":filename,
        "path":destination,"fingerprint":fingerprint,"size_bytes":size,
        "source":{"kind":"huggingface","repo":repo,"file":file,"revision":pinned.revision,"sha256":pinned.sha256},
    }))
}

async fn transfer(
    client: &Client,
    url: Url,
    partial: &Path,
    expected_size: Option<u64>,
    cancel: &AtomicBool,
    progress: &ProgressCallback,
    revision: &str,
) -> Result<()> {
    ensure!(
        !cancel.load(Ordering::Relaxed),
        "download cancelled; partial retained for retry"
    );
    let mut offset = match fs::metadata(partial) {
        Ok(metadata) => {
            ensure!(metadata.is_file(), "download partial is not a regular file");
            metadata.len()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    if let Some(expected) = expected_size {
        ensure!(
            offset <= expected,
            "partial download is larger than the pinned file; inspect or remove {} before retry",
            partial.display()
        );
        if offset == expected {
            return Ok(());
        }
    }
    let mut request = client
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "identity");
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let response = tokio::select! {
        _ = wait_cancelled(cancel) => bail!("download cancelled; partial retained for retry"),
        response = request.send() => response.context("request pinned model bytes")?,
    };
    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
        let complete = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(|value| value.strip_prefix("bytes */"))
            .and_then(|value| value.parse::<u64>().ok());
        ensure!(
            complete == Some(offset) && expected_size.is_none_or(|expected| expected == offset),
            "server rejected resume range; partial size does not match remote file"
        );
        return Ok(());
    }
    check_status(&response)?;
    let (total, response_bytes) = if response.status() == StatusCode::PARTIAL_CONTENT {
        let range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .context("206 response missing Content-Range")?
            .to_str()?;
        let (start, end, total) = parse_content_range(range)?;
        ensure!(
            start == offset,
            "server returned a different resume offset; refusing to corrupt partial model"
        );
        if let Some(expected) = expected_size {
            ensure!(
                expected == total,
                "resume response size differs from pinned metadata"
            );
        }
        let length = end - start + 1;
        if let Some(content_length) = response.content_length() {
            ensure!(
                content_length == length,
                "Content-Length does not match resume range"
            );
        }
        (Some(total), Some(length))
    } else {
        ensure!(
            response.status() == StatusCode::OK,
            "unexpected download response {}",
            response.status()
        );
        // A server may ignore Range. Never append a full response to a partial.
        offset = 0;
        let length = response.content_length();
        if let (Some(expected), Some(length)) = (expected_size, length) {
            ensure!(
                expected == length,
                "response size differs from pinned metadata"
            );
        }
        (expected_size.or(length), length)
    };
    if let Some(total) = total {
        ensure!(
            total <= MAX_MODEL_BYTES,
            "model exceeds 512 GiB download limit"
        );
    }
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create(true);
    if offset == 0 {
        options.truncate(true);
    } else {
        options.append(true);
    }
    let mut file = options
        .open(partial)
        .await
        .context("open partial model download")?;
    let starting_offset = offset;
    let mut stream = response.bytes_stream();
    let mut last_report = Instant::now();
    progress(DownloadProgress {
        downloaded: offset,
        total,
        revision: revision.into(),
        phase: "downloading".into(),
    });
    let transfer_result = async {
        loop {
            let next = tokio::select! {
                _ = wait_cancelled(cancel) => bail!("download cancelled; partial retained for retry"),
                next = stream.next() => next,
            };
            let Some(chunk) = next else { break; };
            let chunk = chunk.context("download interrupted; partial retained for retry")?;
            let new_offset = offset.checked_add(chunk.len() as u64).context("download size overflow")?;
            ensure!(new_offset <= MAX_MODEL_BYTES, "model exceeds 512 GiB download limit");
            if let Some(total) = total { ensure!(new_offset <= total, "server sent more model bytes than declared"); }
            if let Some(length) = response_bytes { ensure!(new_offset - starting_offset <= length, "server sent more bytes than its resume range"); }
            file.write_all(&chunk).await.context("write partial model (check free disk space)")?;
            offset = new_offset;
            if last_report.elapsed() >= Duration::from_millis(150) {
                progress(DownloadProgress { downloaded:offset,total,revision:revision.into(),phase:"downloading".into() });
                last_report = Instant::now();
            }
        }
        if let Some(length) = response_bytes { ensure!(offset - starting_offset == length, "download ended before declared response length; partial retained for retry"); }
        if let Some(total) = total { ensure!(offset == total, "download incomplete; partial retained for retry"); }
        Ok(())
    }.await;
    // Persist successfully received bytes on both ordinary errors and cancellation.
    file.flush().await.context("flush partial download")?;
    file.sync_all().await.context("sync partial download")?;
    progress(DownloadProgress {
        downloaded: offset,
        total,
        revision: revision.into(),
        phase: if transfer_result.is_ok() {
            "downloaded"
        } else {
            "interrupted"
        }
        .into(),
    });
    transfer_result
}

async fn wait_cancelled(cancel: &AtomicBool) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn parse_content_range(value: &str) -> Result<(u64, u64, u64)> {
    let (range, total) = value
        .strip_prefix("bytes ")
        .context("invalid Content-Range unit")?
        .split_once('/')
        .context("invalid Content-Range")?;
    let (start, end) = range
        .split_once('-')
        .context("invalid Content-Range bounds")?;
    let (start, end, total): (u64, u64, u64) = (start.parse()?, end.parse()?, total.parse()?);
    ensure!(
        start <= end && end < total && total <= MAX_MODEL_BYTES,
        "invalid Content-Range bounds"
    );
    Ok((start, end, total))
}

fn check_status(response: &reqwest::Response) -> Result<()> {
    match response.status().as_u16() {
        200..=299 => Ok(()),
        401 | 403 => bail!(
            "Hugging Face access denied ({}). Download gated/private files with your own authorized HF client and import the local GGUF; Codex credentials are never sent to Hugging Face.",
            response.status()
        ),
        404 => bail!("Hugging Face repository, revision, or file was not found"),
        429 => {
            let retry = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("the provider's cooldown");
            bail!(
                "Hugging Face rate limit reached; retry after {retry}. Partial files are retained."
            )
        }
        _ => bail!(
            "Hugging Face returned {}; retry preserves partial files",
            response.status()
        ),
    }
}

fn validate_repo(repo: &str) -> Result<()> {
    let components: Vec<_> = repo.split('/').collect();
    ensure!(
        (1..=2).contains(&components.len())
            && components.iter().all(|part| !part.is_empty()
                && part.len() <= 96
                && *part != "."
                && *part != ".."
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))),
        "repository must be a Hugging Face name or owner/name, not a URL or filesystem path"
    );
    Ok(())
}

fn validate_filename(file: &str) -> Result<()> {
    ensure!(
        !file.is_empty()
            && file.len() <= 1024
            && file.split('/').all(|part| !part.is_empty()
                && part != "."
                && part != ".."
                && !part.chars().any(|c| c.is_control() || c == '\\')),
        "invalid repository-relative model filename"
    );
    Ok(())
}

fn validate_revision_request(revision: &str) -> Result<()> {
    ensure!(
        !revision.is_empty() && revision.len() <= 200 && !revision.chars().any(char::is_control),
        "invalid Hugging Face revision"
    );
    Ok(())
}

fn validate_commit(revision: &str) -> Result<()> {
    ensure!(
        matches!(revision.len(), 40 | 64)
            && revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "Hugging Face revision is not an immutable commit hash"
    );
    Ok(())
}

fn fingerprint_gguf(path: &Path, cancel: Option<&AtomicBool>) -> Result<(String, u64)> {
    let mut file = File::open(path).with_context(|| format!("open model {}", path.display()))?;
    let before = file.metadata()?;
    ensure!(before.is_file(), "model path must refer to a regular file");
    ensure!(
        before.len() >= 8 && before.len() <= MAX_MODEL_BYTES,
        "GGUF size is outside the supported 8-byte to 512-GiB range"
    );
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).context("read GGUF header")?;
    ensure!(
        &prefix[..4] == b"GGUF",
        "file is not a GGUF model (wrong magic bytes)"
    );
    let version = u32::from_le_bytes(prefix[4..].try_into()?);
    ensure!(
        (1..=3).contains(&version),
        "unsupported GGUF version {version}"
    );
    let mut hash = Sha256::new();
    hash.update(prefix);
    let mut bytes_read = 8u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        ensure!(
            cancel.is_none_or(|c| !c.load(Ordering::Relaxed)),
            "model verification cancelled"
        );
        let count = file
            .read(&mut buffer)
            .context("read GGUF for fingerprint")?;
        if count == 0 {
            break;
        }
        bytes_read += count as u64;
        ensure!(
            bytes_read <= MAX_MODEL_BYTES,
            "model grew beyond the size limit while fingerprinting"
        );
        hash.update(&buffer[..count]);
    }
    let after = file.metadata()?;
    ensure!(
        before.len() == bytes_read
            && after.len() == bytes_read
            && before.modified()? == after.modified()?,
        "model changed while fingerprinting; retry when the file is no longer being written"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let named = fs::metadata(path)?;
        ensure!(
            before.dev() == named.dev() && before.ino() == named.ino(),
            "model path was replaced while fingerprinting"
        );
    }
    Ok((hex::encode(hash.finalize()), bytes_read))
}

fn lock_download(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    file.try_lock()
        .context("this model download is already active; wait for it or cancel it first")?;
    Ok(file)
}

fn read_small_file(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= limit as u64,
        "resume metadata exceeds size limit"
    );
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= limit, "resume metadata exceeds size limit");
    Ok(bytes)
}

fn write_metadata_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("{}.tmp", Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(path.parent().context("resume metadata directory")?)?.sync_all()?;
        Ok(())
    })();
    let _ = fs::remove_file(temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::mpsc;

    fn server(response: &'static [u8]) -> Result<(Url, mpsc::Receiver<String>)> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = Url::parse(&format!("http://{}/file.gguf", listener.local_addr()?))?;
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).unwrap_or(0) == 0 {
                    break;
                }
                request.push(byte[0]);
            }
            let _ = sender.send(String::from_utf8_lossy(&request).into_owned());
            let _ = stream.write_all(response);
        });
        Ok((url, receiver))
    }

    fn progress() -> ProgressCallback {
        Arc::new(|_| {})
    }
    fn test_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(5))
            .no_proxy()
            .build()
            .unwrap()
    }

    #[test]
    fn imports_by_reference_and_rejects_drift() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("tiny.gguf");
        fs::write(&path, b"GGUF\x03\0\0\0test")?;
        let model = import_model(&path, Some("Tiny"))?;
        assert_eq!(
            model["path"].as_str().unwrap(),
            path.canonicalize()?.to_str().unwrap()
        );
        assert_eq!(fs::read_dir(root.path())?.count(), 1);
        verify_model(&path, model["fingerprint"].as_str().unwrap())?;
        fs::write(&path, b"GGUF\x03\0\0\0edit")?;
        assert!(verify_model(&path, model["fingerprint"].as_str().unwrap()).is_err());
        fs::write(&path, b"not a model")?;
        assert!(import_model(&path, None).is_err());
        Ok(())
    }

    #[test]
    fn metadata_requires_immutable_revision_and_safe_paths() -> Result<()> {
        let metadata = parse_hf_metadata(
            "org/model",
            &json!({"sha":"a".repeat(40),"siblings":[{"rfilename":"weights.gguf","size":12,"lfs":{"sha256":"b".repeat(64)}},{"rfilename":"README.md"}]}),
        )?;
        assert_eq!(metadata["files"].as_array().unwrap().len(), 1);
        assert!(parse_hf_metadata("org/model", &json!({"sha":"main","siblings":[]})).is_err());
        assert!(validate_repo("https://evil.invalid/file").is_err());
        assert!(validate_filename("../../model.gguf").is_err());
        assert!(parse_content_range("bytes 8-7/12").is_err());
        assert!(parse_content_range("bytes 4-12/12").is_err());
        Ok(())
    }

    #[tokio::test]
    async fn partial_content_resumes_exactly() -> Result<()> {
        let root = tempfile::tempdir()?;
        let partial = root.path().join("model.part");
        fs::write(&partial, b"GGUF")?;
        let (url, request) = server(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 4-11/12\r\nContent-Length: 8\r\nConnection: close\r\n\r\n\x03\0\0\0test")?;
        transfer(
            &test_client(),
            url,
            &partial,
            Some(12),
            &AtomicBool::new(false),
            &progress(),
            "revision",
        )
        .await?;
        assert!(request.recv()?.contains("range: bytes=4-"));
        assert_eq!(fs::read(&partial)?, b"GGUF\x03\0\0\0test");
        Ok(())
    }

    #[tokio::test]
    async fn ignored_range_restarts_without_appending() -> Result<()> {
        let root = tempfile::tempdir()?;
        let partial = root.path().join("model.part");
        fs::write(&partial, b"old partial")?;
        let (url, _) = server(
            b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\nGGUF\x03\0\0\0test",
        )?;
        transfer(
            &test_client(),
            url,
            &partial,
            Some(12),
            &AtomicBool::new(false),
            &progress(),
            "revision",
        )
        .await?;
        assert_eq!(fs::read(&partial)?, b"GGUF\x03\0\0\0test");
        Ok(())
    }

    #[tokio::test]
    async fn mismatched_range_preserves_partial_and_fails() -> Result<()> {
        let root = tempfile::tempdir()?;
        let partial = root.path().join("model.part");
        fs::write(&partial, b"GGUF")?;
        let (url, _) = server(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 5-11/12\r\nContent-Length: 7\r\nConnection: close\r\n\r\n\0\0\0test")?;
        assert!(
            transfer(
                &test_client(),
                url,
                &partial,
                Some(12),
                &AtomicBool::new(false),
                &progress(),
                "revision"
            )
            .await
            .is_err()
        );
        assert_eq!(fs::read(&partial)?, b"GGUF");
        Ok(())
    }

    #[tokio::test]
    async fn interrupted_stream_can_be_resumed() -> Result<()> {
        let root = tempfile::tempdir()?;
        let partial = root.path().join("model.part");
        let (url, _) = server(
            b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\nGGUF\x03\0\0\0",
        )?;
        assert!(
            transfer(
                &test_client(),
                url,
                &partial,
                Some(12),
                &AtomicBool::new(false),
                &progress(),
                "revision"
            )
            .await
            .is_err()
        );
        assert_eq!(fs::read(&partial)?, b"GGUF\x03\0\0\0");
        let (url, request) = server(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 8-11/12\r\nContent-Length: 4\r\nConnection: close\r\n\r\ntest")?;
        transfer(
            &test_client(),
            url,
            &partial,
            Some(12),
            &AtomicBool::new(false),
            &progress(),
            "revision",
        )
        .await?;
        assert!(request.recv()?.contains("range: bytes=8-"));
        assert_eq!(fs::read(&partial)?, b"GGUF\x03\0\0\0test");
        Ok(())
    }

    #[tokio::test]
    async fn complete_partial_and_cancel_do_not_fetch() -> Result<()> {
        let root = tempfile::tempdir()?;
        let partial = root.path().join("model.part");
        fs::write(&partial, b"GGUF\x03\0\0\0test")?;
        let url = Url::parse("http://127.0.0.1:1/unreachable")?;
        transfer(
            &test_client(),
            url.clone(),
            &partial,
            Some(12),
            &AtomicBool::new(false),
            &progress(),
            "revision",
        )
        .await?;
        assert!(
            transfer(
                &test_client(),
                url,
                &partial,
                Some(12),
                &AtomicBool::new(true),
                &progress(),
                "revision"
            )
            .await
            .is_err()
        );
        assert_eq!(fs::read(&partial)?.len(), 12);
        Ok(())
    }

    #[tokio::test]
    async fn access_and_rate_limit_errors_preserve_received_bytes() -> Result<()> {
        for (response, expected_message) in [
            (&b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"[..], "access denied"),
            (&b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 60\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"[..], "retry after 60"),
        ] {
            let root = tempfile::tempdir()?;
            let partial = root.path().join("model.part");
            fs::write(&partial, b"GGUF")?;
            let (url, _) = server(response)?;
            let error = transfer(&test_client(), url, &partial, Some(12), &AtomicBool::new(false), &progress(), "revision").await.unwrap_err();
            assert!(error.to_string().contains(expected_message), "{error}");
            assert_eq!(fs::read(&partial)?, b"GGUF");
        }
        Ok(())
    }

    #[tokio::test]
    async fn unsatisfiable_range_only_accepts_exact_complete_file() -> Result<()> {
        let root = tempfile::tempdir()?;
        let partial = root.path().join("model.part");
        fs::write(&partial, b"GGUF\x03\0\0\0test")?;
        let (url, _) = server(b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */12\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
        transfer(
            &test_client(),
            url,
            &partial,
            None,
            &AtomicBool::new(false),
            &progress(),
            "revision",
        )
        .await?;
        let (url, _) = server(b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */13\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
        assert!(
            transfer(
                &test_client(),
                url,
                &partial,
                None,
                &AtomicBool::new(false),
                &progress(),
                "revision"
            )
            .await
            .is_err()
        );
        assert_eq!(fs::read(&partial)?.len(), 12);
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_interrupts_a_stalled_body_and_keeps_partial() -> Result<()> {
        let root = tempfile::tempdir()?;
        let partial = root.path().join("model.part");
        fs::write(&partial, b"GGUF")?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = Url::parse(&format!("http://{}/model", listener.local_addr()?))?;
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0; 1];
            while !request.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).unwrap_or(0) == 0 {
                    return;
                }
                request.push(byte[0]);
            }
            let _ = stream.write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 4-11/12\r\nContent-Length: 8\r\nConnection: close\r\n\r\n\x03\0\0\0");
            std::thread::sleep(Duration::from_millis(500));
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            trigger.store(true, Ordering::Relaxed);
        });
        let error = tokio::time::timeout(
            Duration::from_secs(2),
            transfer(
                &test_client(),
                url,
                &partial,
                Some(12),
                &cancel,
                &progress(),
                "revision",
            ),
        )
        .await?
        .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        let received = fs::read(&partial)?;
        assert!(received.starts_with(b"GGUF") && received.len() <= 8);
        Ok(())
    }
}
