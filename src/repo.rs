use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io::BufReader,
    path::PathBuf,
};

use color_eyre::eyre::{Context, Result};
use futures::future::try_join_all;
use hf_hub::{
    api::tokio::{Api, ApiError, ApiRepo, Progress},
    Cache, Repo,
};
use indicatif::{MultiProgress, ProgressBar};
use reqwest::StatusCode;
use serde::Deserialize;

use crate::config::Config;

#[derive(Debug, Deserialize)]
struct Index {
    pub weight_map: HashMap<String, String>,
}

pub struct SafeTensorsRepo {
    repo: Repo,
    api_repo: ApiRepo,
}

impl SafeTensorsRepo {
    pub fn new(api: &Api, repo: Repo) -> Self {
        let api_repo = api.repo(repo.clone());
        Self { repo, api_repo }
    }

    pub async fn get_checkpoint_paths(&self) -> Result<Vec<PathBuf>> {
        let checkpoints = self.get_safetensors_index().await?;

        let multi_pg = MultiProgress::new();

        let mut tasks = Vec::new();
        for checkpoint in checkpoints {
            let pb_task = multi_pg.add(ProgressBar::new(100));
            tasks.push(self.get_with_progress(checkpoint, pb_task));
        }

        Ok(try_join_all(tasks).await?)
    }

    pub async fn get_config(&self) -> Result<Config> {
        let config_file = self.api_repo.get("config.json").await?;
        let reader = BufReader::new(File::open(&config_file).context(format!(
            "Cannot open model configuration for reading: {}",
            config_file.to_string_lossy()
        ))?);
        Ok(serde_json::from_reader(reader)?)
    }

    async fn get_with_progress<P>(
        &self,
        filename: impl AsRef<str>,
        progress: P,
    ) -> Result<PathBuf, ApiError>
    where
        P: Progress + Clone + Send + Sync + 'static,
    {
        let filename = filename.as_ref();
        let cache = Cache::from_env();
        if let Some(path) = cache.repo(self.repo.clone()).get(filename) {
            Ok(path)
        } else {
            self.api_repo
                .download_with_progress(filename, progress)
                .await
        }
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
        let checkpoint_set = index.weight_map.into_values().collect::<BTreeSet<_>>();
        Ok(checkpoint_set.into_iter().collect())
    }
}
