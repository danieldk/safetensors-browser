use std::{
    collections::{HashMap, HashSet},
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
use serde::Deserialize;

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
        let index = self.get_safetensors_index().await?;

        let multi_pg = MultiProgress::new();

        let checkpoints: HashSet<&String> = HashSet::from_iter(index.weight_map.values());

        let mut tasks = Vec::new();
        for checkpoint in checkpoints {
            let pb_task = multi_pg.add(ProgressBar::new(100));
            tasks.push(self.get_with_progress(checkpoint, pb_task));
        }

        Ok(try_join_all(tasks).await?)
    }

    async fn get_with_progress<P>(&self, filename: &str, progress: P) -> Result<PathBuf, ApiError>
    where
        P: Progress + Clone + Send + Sync + 'static,
    {
        let cache = Cache::from_env();
        if let Some(path) = cache.repo(self.repo.clone()).get(filename) {
            Ok(path)
        } else {
            self.api_repo
                .download_with_progress(filename, progress)
                .await
        }
    }

    async fn get_safetensors_index(&self) -> Result<Index> {
        let index_file = self.api_repo.get("model.safetensors.index.json").await?;
        let reader = BufReader::new(File::open(&index_file).context(format!(
            "Cannot open safetensors index for reading: {}",
            index_file.to_string_lossy()
        ))?);
        Ok(serde_json::from_reader(reader)?)
    }
}
