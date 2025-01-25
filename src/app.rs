use std::collections::HashMap;

use color_eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{
        palette::tailwind::{BLUE, SLATE},
        Color, Modifier, Style, Stylize,
    },
    symbols,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListState, Padding, Paragraph, StatefulWidget, Widget, Wrap},
    DefaultTerminal, Frame,
};
use safetensors::tensor::TensorInfo;

const TODO_HEADER_STYLE: Style = Style::new().fg(SLATE.c100).bg(BLUE.c800);
const NORMAL_ROW_BG: Color = SLATE.c950;
const ALT_ROW_BG_COLOR: Color = SLATE.c900;
const SELECTED_STYLE: Style = Style::new().bg(SLATE.c800).add_modifier(Modifier::BOLD);
const TEXT_FG_COLOR: Color = SLATE.c200;

#[derive(Debug, Default)]
pub struct App {
    tensor_names: Vec<String>,
    tensors: HashMap<String, TensorInfo>,
    tensor_state: ListState,
    running: bool,
}

impl App {
    /// Construct a new instance of [`App`].
    pub fn new(tensors: HashMap<String, TensorInfo>) -> Self {
        Self {
            tensor_names: tensors.keys().map(ToOwned::to_owned).collect(),
            tensors,
            tensor_state: Default::default(),
            running: false,
        }
    }

    /// Run the application's main loop.
    pub fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        self.running = true;
        while self.running {
            terminal.draw(|frame| frame.render_widget(&mut self, frame.area()))?;
            if let Event::Key(key) = event::read()? {
                self.handle_key(key);
            };
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit(),
            //KeyCode::Char('h') | KeyCode::Left => self.select_none(),
            KeyCode::Char('j') | KeyCode::Down => self.select_next(),
            KeyCode::Char('k') | KeyCode::Up => self.select_previous(),
            KeyCode::Char('g') | KeyCode::Home => self.select_first(),
            KeyCode::Char('G') | KeyCode::End => self.select_last(),
            //KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
            //    self.toggle_status();
            // }
            _ => {}
        }
    }

    fn render_list(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::new()
            .title(Line::raw("Tensors").centered())
            .borders(Borders::TOP)
            .border_set(symbols::border::EMPTY)
            .border_style(TODO_HEADER_STYLE)
            .bg(NORMAL_ROW_BG);

        let tensors = List::new(self.tensor_names.iter().map(String::as_str))
            .block(block)
            .highlight_style(SELECTED_STYLE)
            .highlight_symbol(">")
            .highlight_spacing(ratatui::widgets::HighlightSpacing::Always);
        StatefulWidget::render(tensors, area, buf, &mut self.tensor_state);
    }

    fn render_selected_item(&mut self, area: Rect, buf: &mut Buffer) {
        let info = if let Some(i) = self.tensor_state.selected() {
            let name = &self.tensor_names[i];
            let tensor_info = &self.tensors[name];
            let field_style = Style::new().magenta();
            vec![
                Line::from(vec![Span::styled("Name: ", field_style), Span::raw(name)]),
                Line::from(vec![
                    Span::styled("DType: ", field_style),
                    Span::raw(format!("{:?}", tensor_info.dtype)),
                ]),
                Line::from(vec![
                    Span::styled("Shape: ", field_style),
                    Span::raw(format!("{:?}", tensor_info.shape)),
                ]),
                Line::from(vec![
                    Span::styled("Offsets: ", field_style),
                    Span::raw(format!("{:?}", tensor_info.data_offsets)),
                ]),
            ]
        } else {
            vec![Line::raw("Nothing selected...")]
            //"Nothing selected...".to_string()
        };

        // We show the list item's info under the list in this paragraph
        let block = Block::new()
            .title(Line::raw("Metadata").centered())
            .borders(Borders::TOP)
            .border_set(symbols::border::EMPTY)
            .border_style(TODO_HEADER_STYLE)
            .bg(NORMAL_ROW_BG)
            .padding(Padding::horizontal(1));

        // We can now render the item info
        Paragraph::new(info)
            .block(block)
            .fg(TEXT_FG_COLOR)
            .wrap(Wrap { trim: false })
            .render(area, buf);
    }

    fn quit(&mut self) {
        self.running = false;
    }

    fn select_first(&mut self) {
        self.tensor_state.select_first();
    }

    fn select_last(&mut self) {
        self.tensor_state.select_last();
    }

    fn select_next(&mut self) {
        self.tensor_state.select_next();
    }

    fn select_previous(&mut self) {
        self.tensor_state.select_previous();
    }
}

impl Widget for &mut App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [header_area, main_area, footer_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(area);

        let [list_area, item_area] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Fill(1)]).areas(main_area);

        self.render_list(list_area, buf);
        self.render_selected_item(item_area, buf);
    }
}
