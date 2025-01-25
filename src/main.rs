use std::{collections::HashMap, fs::File, path::Path};

use clap::Parser;
use color_eyre::eyre::{Context, Result};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use memmap2::Mmap;
use safetensors::{tensor::TensorInfo, SafeTensors};
use tokio::runtime::Runtime;

pub mod app;
pub use app::App;

pub mod metadata;

mod repo;
use repo::SafeTensorsRepo;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    repo: String,

    #[arg(short, long)]
    revision: Option<String>,
}

fn get_tensors<P>(checkpoint_paths: &[P]) -> Result<HashMap<String, TensorInfo>>
where
    P: AsRef<Path>,
{
    let mut tensors = HashMap::new();
    for path in checkpoint_paths {
        let path = path.as_ref();
        let f = File::open(&path)
            .with_context(|| format!("Cannot open checkpoint: {}", path.to_string_lossy()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        let (_, metadata) = SafeTensors::read_metadata(&mmap)?;
        tensors.extend(
            metadata
                .tensors()
                .into_iter()
                .map(|(name, info)| (name, info.clone())),
        );
    }

    Ok(tensors)
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    let api = Api::new()?;
    let revision = cli.revision.unwrap_or_else(|| "main".to_string());
    let repo = Repo::with_revision(cli.repo, RepoType::Model, revision);
    let safetensors_repo = SafeTensorsRepo::new(&api, repo);
    let rt = Runtime::new()?;
    let checkpoint_paths = rt.block_on(safetensors_repo.get_checkpoint_paths())?;

    let terminal = ratatui::init();
    let result = App::new(get_tensors(&checkpoint_paths)?).run(terminal);
    ratatui::restore();

    result
}
