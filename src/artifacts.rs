//! Bounded, content-addressed artifacts. Database rows may reference a blob only
//! after its atomic, durable installation has completed.

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

pub const MAX_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_TENSOR_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_TENSOR_RANK: usize = 8;
const TENSOR_MAGIC: &[u8; 8] = b"TNF32\0\x01\0";

#[derive(Clone, Debug)]
pub struct ArtifactStore {
    root: Arc<PathBuf>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl ArtifactStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(root.as_ref()).context("create artifact directory")?;
        Ok(Self {
            root: Arc::new(root.as_ref().canonicalize()?),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, hash: &str) -> Result<PathBuf> {
        validate_hash(hash)?;
        Ok(self.root.join(&hash[..2]).join(hash))
    }

    /// SHA-256 covers the complete file, including a tensor's shape and header.
    pub fn put_bytes(&self, bytes: &[u8]) -> Result<String> {
        ensure!(
            bytes.len() <= MAX_ARTIFACT_BYTES,
            "artifact exceeds 128 MiB limit"
        );
        let hash = sha256(bytes);
        let path = self.path(&hash)?;
        if path.try_exists()? {
            self.read_bytes(&hash)?; // Do not silently repair or accept corruption.
            return Ok(hash);
        }
        fs::create_dir_all(path.parent().context("artifact parent")?)?;
        // A new hash shard is itself a directory entry; make it durable too.
        File::open(self.root())?.sync_all()?;
        install_new_atomic(&path, bytes)?;
        // A concurrent writer may have won installation. Verify its bytes too.
        self.read_bytes(&hash)?;
        Ok(hash)
    }

    pub fn import_bytes(&self, expected_hash: &str, bytes: &[u8]) -> Result<String> {
        validate_hash(expected_hash)?;
        ensure!(
            bytes.len() <= MAX_ARTIFACT_BYTES,
            "artifact exceeds 128 MiB limit"
        );
        ensure!(sha256(bytes) == expected_hash, "artifact checksum mismatch");
        self.put_bytes(bytes)
    }

    pub fn read_bytes(&self, hash: &str) -> Result<Vec<u8>> {
        let path = self.path(hash)?;
        let file = File::open(&path).with_context(|| format!("open artifact {hash}"))?;
        ensure!(file.metadata()?.is_file(), "artifact is not a regular file");
        ensure!(
            file.metadata()?.len() <= MAX_ARTIFACT_BYTES as u64,
            "artifact exceeds 128 MiB limit"
        );
        let mut bytes = Vec::new();
        file.take(MAX_ARTIFACT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= MAX_ARTIFACT_BYTES,
            "artifact exceeds 128 MiB limit"
        );
        ensure!(
            sha256(&bytes) == hash,
            "artifact checksum mismatch for {hash}"
        );
        Ok(bytes)
    }

    pub fn put_json(&self, value: &Value) -> Result<String> {
        self.put_bytes(&serde_json::to_vec(value)?)
    }

    pub fn read_json(&self, hash: &str) -> Result<Value> {
        serde_json::from_slice(&self.read_bytes(hash)?).context("artifact is not valid JSON")
    }

    pub fn put_jsonl(&self, records: &[Value]) -> Result<String> {
        let mut bytes = Vec::new();
        for record in records {
            serde_json::to_writer(&mut bytes, record)?;
            bytes.push(b'\n');
            ensure!(
                bytes.len() <= MAX_ARTIFACT_BYTES,
                "JSONL artifact exceeds 128 MiB limit"
            );
        }
        self.put_bytes(&bytes)
    }

    pub fn read_jsonl(&self, hash: &str) -> Result<Vec<Value>> {
        let bytes = self.read_bytes(hash)?;
        bytes
            .split(|b| *b == b'\n')
            .enumerate()
            .filter(|(_, line)| !line.is_empty())
            .map(|(index, line)| {
                serde_json::from_slice(line)
                    .with_context(|| format!("invalid JSONL artifact line {}", index + 1))
            })
            .collect()
    }

    pub fn put_f32(&self, shape: &[usize], values: &[f32]) -> Result<String> {
        ensure!(
            tensor_elements(shape)? == values.len(),
            "tensor shape does not match value count"
        );
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "tensor contains non-finite values"
        );
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(12 + 8 * shape.len() + 4 * values.len())
            .context("allocate tensor artifact")?;
        bytes.extend_from_slice(TENSOR_MAGIC);
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.put_bytes(&bytes)
    }

    pub fn read_f32(&self, hash: &str) -> Result<Tensor> {
        decode_tensor(&self.read_bytes(hash)?)
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn validate_hash(hash: &str) -> Result<()> {
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "expected lowercase SHA-256 hex digest"
    );
    Ok(())
}

fn tensor_elements(shape: &[usize]) -> Result<usize> {
    ensure!(
        !shape.is_empty() && shape.len() <= MAX_TENSOR_RANK,
        "tensor rank must be 1..=8"
    );
    let mut count = 1usize;
    for dimension in shape {
        ensure!(*dimension > 0, "tensor dimensions must be positive");
        count = count
            .checked_mul(*dimension)
            .context("tensor shape overflow")?;
        ensure!(count <= MAX_TENSOR_ELEMENTS, "tensor exceeds element limit");
    }
    Ok(count)
}

fn decode_tensor(bytes: &[u8]) -> Result<Tensor> {
    ensure!(
        bytes.len() >= 12 && &bytes[..8] == TENSOR_MAGIC,
        "invalid or unsupported F32 tensor header"
    );
    let rank = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    ensure!(
        (1..=MAX_TENSOR_RANK).contains(&rank),
        "tensor rank must be 1..=8"
    );
    let header = 12 + rank * 8;
    ensure!(bytes.len() >= header, "truncated tensor shape");
    let mut shape = Vec::with_capacity(rank);
    for dim in bytes[12..header].chunks_exact(8) {
        shape.push(usize::try_from(u64::from_le_bytes(dim.try_into()?))?);
    }
    let count = tensor_elements(&shape)?;
    ensure!(
        bytes.len() == header + count * 4,
        "tensor byte count does not match shape"
    );
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .context("allocate tensor values")?;
    for chunk in bytes[header..].chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into()?);
        ensure!(value.is_finite(), "tensor contains non-finite values");
        values.push(value);
    }
    Ok(Tensor { shape, values })
}

/// Install without replacing any existing blob. The temporary file is on the
/// same filesystem, and linking exposes only fully written, fsynced contents.
fn install_new_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("artifact parent")?;
    let temporary = parent.join(format!(".{}.tmp", Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        match fs::hard_link(&temporary, path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => bail!("install artifact: {error}"),
        }
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip_json_jsonl_and_little_endian_tensor() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = ArtifactStore::open(root.path())?;
        let value = json!({"answer": 42});
        let hash = store.put_json(&value)?;
        assert_eq!(store.read_json(&hash)?, value);
        assert_eq!(hash, store.put_json(&value)?);
        let records = vec![value.clone(), json!({"message":"with\na newline"})];
        assert_eq!(store.read_jsonl(&store.put_jsonl(&records)?)?, records);
        let hash = store.put_f32(&[2, 2], &[1.0, -2.0, 0.0, 3.5])?;
        assert_eq!(&store.read_bytes(&hash)?[28..32], &1f32.to_le_bytes());
        assert_eq!(
            store.read_f32(&hash)?,
            Tensor {
                shape: vec![2, 2],
                values: vec![1., -2., 0., 3.5]
            }
        );
        Ok(())
    }

    #[test]
    fn corruption_hash_and_shape_fail_closed() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = ArtifactStore::open(root.path())?;
        let hash = store.put_f32(&[2], &[1., 2.])?;
        let bytes = store.read_bytes(&hash)?;
        assert!(store.import_bytes(&"0".repeat(64), &bytes).is_err());
        fs::write(store.path(&hash)?, b"corrupt")?;
        assert!(store.read_f32(&hash).is_err());
        assert!(store.put_bytes(&bytes).is_err());
        assert!(store.read_bytes("../../private-file").is_err());
        assert!(store.put_f32(&[2], &[1.]).is_err());
        assert!(store.put_f32(&[usize::MAX, 2], &[]).is_err());
        assert!(store.put_f32(&[1], &[f32::NAN]).is_err());
        assert!(store.put_f32(&[1], &[f32::INFINITY]).is_err());
        let mut invalid = bytes;
        invalid[12..20].copy_from_slice(&3u64.to_le_bytes());
        let malformed = store.put_bytes(&invalid)?;
        assert!(store.read_f32(&malformed).is_err());
        invalid[12..20].copy_from_slice(&2u64.to_le_bytes());
        invalid[20..24].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(store.read_f32(&store.put_bytes(&invalid)?).is_err());
        Ok(())
    }

    #[test]
    fn concurrent_install_is_immutable() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = ArtifactStore::open(root.path())?;
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let store = store.clone();
                std::thread::spawn(move || store.put_bytes(b"identical payload"))
            })
            .collect();
        for thread in handles {
            let hash = thread.join().unwrap()?;
            assert_eq!(store.read_bytes(&hash)?, b"identical payload");
        }
        Ok(())
    }
}
