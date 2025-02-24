use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use bytes::Buf;
use color_eyre::eyre::{Context, Result};
use futures::future::try_join_all;
use hf_hub::api::tokio::{Api, ApiError, ApiRepo};
use hf_hub::{CacheRepo, Repo};
use indicatif::ProgressBar;
use reqwest::StatusCode;
use safetensors::tensor::Metadata;
use serde::Deserialize;
use tempfile::NamedTempFile;

use crate::config::Config;
use crate::utils::symlink_or_rename;

const MAX_CONCURRENT: usize = 8;

#[derive(Debug, Deserialize)]
struct Index {
    pub weight_map: HashMap<String, String>,
}

pub struct CheckpointMetadata {
    pub filename: String,
    pub metadata: Metadata,
}

pub struct SafeTensorsRepo {
    api: Api,
    api_repo: ApiRepo,
    cache_repo: CacheRepo,
}

impl SafeTensorsRepo {
    pub fn new(api: &Api, repo: Repo, cache_repo: CacheRepo) -> Self {
        let api_repo = api.repo(repo.clone());
        Self {
            api: api.clone(),
            api_repo,
            cache_repo,
        }
    }

    async fn download_file(&self, filename: &str) -> Result<CheckpointMetadata> {
        let url = self.api_repo.url(filename);
        let url_metadata = self.api.metadata(&url).await?;
        let blob_path = self.cache_repo.blob_path(&url_metadata.etag);

        let client = self.api.client();
        let header_size_response = client.get(&url).header("RANGE", "bytes=0-7").send().await?;
        let mut header_size_bytes = header_size_response.bytes().await?;
        let header_size = header_size_bytes.get_u64_le();

        let metadata_response = client
            .get(&url)
            .header("RANGE", format!("bytes=8-{}", header_size + 7))
            .send()
            .await?;

        let metadata_bytes = metadata_response.bytes().await?.to_vec();
        let metadata: Metadata = serde_json::from_slice(&metadata_bytes)?;

        let mut tempfile = NamedTempFile::new()?;

        {
            let mut writer = BufWriter::new(&mut tempfile);
            writer.write_all(&header_size.to_le_bytes())?;
            writer.write_all(&metadata_bytes)?;
        }

        std::fs::create_dir_all(blob_path.parent().unwrap())?;
        std::fs::copy(tempfile.path(), &blob_path)?;

        let mut pointer_path = self.cache_repo.pointer_path(&url_metadata.commit_hash);
        std::fs::create_dir_all(&pointer_path)?;
        pointer_path.push(filename);
        symlink_or_rename(&blob_path, &pointer_path)?;
        self.cache_repo.create_ref(&url_metadata.commit_hash)?;

        Ok(CheckpointMetadata {
            filename: filename.to_owned(),
            metadata,
        })
    }

    fn file_from_cache(&self, filename: &str) -> Result<Option<CheckpointMetadata>> {
        let metadata_path = match self.cache_repo.get(filename) {
            Some(path) => path,
            None => return Ok(None),
        };

        let f = File::open(metadata_path)?;
        let mut reader = BufReader::new(f);
        let mut metadata_length_bytes = [0; 8];
        match reader.read_exact(&mut metadata_length_bytes) {
            Ok(_) => (),
            Err(_) => return Ok(None),
        };

        let metadata_length = u64::from_le_bytes(metadata_length_bytes);
        let mut metadata_bytes = vec![0; metadata_length as usize];
        match reader.read_exact(&mut metadata_bytes) {
            Ok(_) => (),
            Err(_) => return Ok(None),
        }

        match serde_json::from_slice(&metadata_bytes) {
            Ok(metadata) => Ok(Some(CheckpointMetadata {
                metadata,
                filename: filename.to_owned(),
            })),
            Err(_) => Ok(None),
        }
    }

    async fn get_file(
        &self,
        filename: String,
        progress: &ProgressBar,
    ) -> Result<CheckpointMetadata> {
        let metadata = match self.file_from_cache(&filename)? {
            Some(metadata) => metadata,
            None => self.download_file(&filename).await?,
        };
        progress.inc(1);
        Ok(metadata)
    }

    pub async fn get_checkpoint_metadatas(&self) -> Result<Vec<CheckpointMetadata>> {
        let checkpoints = self.get_safetensors_index().await?;

        let progress = ProgressBar::new(checkpoints.len() as u64);
        progress.tick();

        let info = self.api_repo.info().await?;
        self.cache_repo.create_ref(&info.sha)?;

        let mut results = Vec::new();
        let mut tasks = Vec::new();
        for checkpoint in checkpoints {
            tasks.push(self.get_file(checkpoint, &progress));

            if tasks.len() == MAX_CONCURRENT {
                results.extend(try_join_all(tasks).await?);
                tasks = Vec::new();
            }
        }

        if !tasks.is_empty() {
            results.extend(try_join_all(tasks).await?);
        }

        progress.finish();

        Ok(results)
    }

    pub async fn get_config(&self) -> Result<Config> {
        let config_file = self.api_repo.get("config.json").await?;
        let reader = BufReader::new(File::open(&config_file).context(format!(
            "Cannot open model configuration for reading: {}",
            config_file.to_string_lossy()
        ))?);
        Ok(serde_json::from_reader(reader)?)
    }

    async fn get_safetensors_index(&self) -> Result<Vec<String>> {
        let index_file = match self.api_repo.get("model.safetensors.index.json").await {
            Ok(index_file) => Ok(index_file),
            Err(ApiError::RequestError(request_err)) => {
                if let Some(StatusCode::NOT_FOUND) = request_err.status() {
                    return Ok(vec!["model.safetensors".to_string()]);
                }
                Err(ApiError::RequestError(request_err))
            }
            err => err,
        }?;
        let reader = BufReader::new(File::open(&index_file).context(format!(
            "Cannot open safetensors index for reading: {}",
            index_file.to_string_lossy()
        ))?);

        let index: Index = serde_json::from_reader(reader)?;
        let checkpoint_set = index.weight_map.into_values().collect::<HashSet<_>>();
        Ok(checkpoint_set.into_iter().collect())
    }
}
