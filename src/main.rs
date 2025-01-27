use clap::Parser;
use color_eyre::eyre::Result;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use models::get_param_layer;
use tokio::runtime::Runtime;
use tokio::try_join;

pub mod app;
pub use app::App;

pub mod config;

mod input;
pub use input::InputState;

pub mod metadata;
use metadata::get_tensors;

mod repo;
use repo::SafeTensorsRepo;

pub mod models;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    repo: String,

    #[arg(short, long)]
    revision: Option<String>,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    let api = Api::new()?;
    let revision = cli.revision.unwrap_or_else(|| "main".to_string());
    let repo = Repo::with_revision(cli.repo, RepoType::Model, revision);
    let safetensors_repo = SafeTensorsRepo::new(&api, repo);
    let rt = Runtime::new()?;
    let (checkpoint_paths, config) = rt.block_on(async {
        try_join!(
            safetensors_repo.get_checkpoint_paths(),
            safetensors_repo.get_config()
        )
    })?;

    let terminal = ratatui::init();
    let param_to_layer = get_param_layer(&config.model_type);
    let result = App::new(get_tensors(
        &checkpoint_paths,
        config.layer_quantizers(),
        param_to_layer.as_deref(),
    )?)
    .run(terminal);
    ratatui::restore();

    result
}
