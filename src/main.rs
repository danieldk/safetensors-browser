use clap::Parser;
use color_eyre::eyre::Result;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokio::runtime::Runtime;

pub mod app;
pub use app::App;

mod input;
pub use input::InputState;

pub mod metadata;
use metadata::get_tensors;

mod repo;
use repo::SafeTensorsRepo;

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
    let checkpoint_paths = rt.block_on(safetensors_repo.get_checkpoint_paths())?;

    let terminal = ratatui::init();
    let result = App::new(get_tensors(&checkpoint_paths)?).run(terminal);
    ratatui::restore();

    result
}
