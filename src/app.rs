use std::collections::HashMap;

use color_eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Position, Rect},
    style::{
        palette::{
            material::PINK,
            tailwind::{BLUE, SLATE},
        },
        Color, Modifier, Style, Stylize,
    },
    symbols,
    text::Line,
    widgets::{
        Block, Borders, List, ListState, Padding, Paragraph, Scrollbar, ScrollbarState,
        StatefulWidget, Widget, Wrap,
    },
    DefaultTerminal,
};

use crate::{
    metadata::{cmp_numeric_lexicographic, RenderMetadata, TensorMetadata},
    InputState,
};

const TODO_HEADER_STYLE: Style = Style::new().fg(SLATE.c100).bg(BLUE.c800);
const NORMAL_ROW_BG: Color = SLATE.c950;
const SELECTED_STYLE: Style = Style::new().bg(PINK.c600).add_modifier(Modifier::BOLD);
const TEXT_FG_COLOR: Color = SLATE.c200;

#[derive(Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
enum UiState {
    #[default]
    Browse,
    Filter,
    Init,
    Quit,
}

pub struct App {
    cursor_position: Option<Position>,
    matcher: SkimMatcherV2,
    tensor_names: Vec<String>,
    tensors: HashMap<String, TensorMetadata>,
    tensor_list: List<'static>,
    tensor_state: ListState,
    tensor_scrollbar_state: ScrollbarState,
    state: UiState,
    filter_state: InputState,
}

impl App {
    /// Construct a new instance of [`App`].
    pub fn new(tensors: HashMap<String, TensorMetadata>) -> Self {
        let scroll_len = tensors.len();

        Self {
            cursor_position: None,
            filter_state: Default::default(),
            matcher: Default::default(),
            tensor_list: Default::default(),
            tensor_names: Default::default(),
            tensors,
            tensor_state: Default::default(),
            tensor_scrollbar_state: ScrollbarState::new(scroll_len),
            state: UiState::Init,
        }
    }

    /// Run the application's main loop.
    pub fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        while !matches!(self.state, UiState::Quit) {
            if matches!(self.state, UiState::Init | UiState::Filter) {
                self.update_tensor_names();
            }

            if matches!(self.state, UiState::Init) {
                self.state = UiState::Browse;
            }

            terminal.draw(|frame| {
                frame.render_widget(&mut self, frame.area());
                if let Some(cursor_position) = self.cursor_position {
                    frame.set_cursor_position(cursor_position);
                }
            })?;
            if let Event::Key(key) = event::read()? {
                self.handle_key(key);
            };
        }
        Ok(())
    }

    fn update_tensor_names(&mut self) {
        let new_tensor_names: Vec<String> = self
            .tensors
            .keys()
            .filter(|name| {
                self.matcher
                    .fuzzy_match(name, self.filter_state.text())
                    .is_some()
            })
            .map(String::clone)
            .collect();

        if new_tensor_names.len() == self.tensor_names.len() {
            return;
        }

        self.tensor_names = new_tensor_names;

        self.tensor_names
            .sort_by(|k1, k2| cmp_numeric_lexicographic(k1, k2));
        self.tensor_scrollbar_state = self
            .tensor_scrollbar_state
            .content_length(self.tensor_names.len());

        self.tensor_list = List::new(self.tensor_names.iter().map(ToOwned::to_owned))
            .highlight_style(SELECTED_STYLE)
            .highlight_symbol(">")
            .highlight_spacing(ratatui::widgets::HighlightSpacing::Always);

        self.tensor_state.select_first();
        self.tensor_scrollbar_state.first();
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        match self.state {
            UiState::Browse => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => self.quit(),
                //KeyCode::Char('h') | KeyCode::Left => self.select_none(),
                KeyCode::Char('j') | KeyCode::Down => self.select_next(),
                KeyCode::Char('k') | KeyCode::Up => self.select_previous(),
                KeyCode::Char('g') | KeyCode::Home => self.select_first(),
                KeyCode::Char('G') | KeyCode::End => self.select_last(),
                KeyCode::Char('/') => self.state = UiState::Filter,
                KeyCode::PageDown => self.page_down(),
                KeyCode::PageUp => self.page_up(),
                _ => {}
            },
            UiState::Filter => match key.code {
                KeyCode::Enter => self.state = UiState::Browse,
                KeyCode::Esc => self.state = UiState::Browse,
                KeyCode::Char(to_insert) => self.filter_state.enter_char(to_insert),
                KeyCode::Backspace => self.filter_state.delete_char(),
                KeyCode::Left => self.filter_state.move_cursor_left(),
                KeyCode::Right => self.filter_state.move_cursor_right(),
                KeyCode::Down => self.select_next(),
                KeyCode::Up => self.select_previous(),
                KeyCode::Home => self.select_first(),
                KeyCode::End => self.select_last(),
                KeyCode::PageDown => self.page_down(),
                KeyCode::PageUp => self.page_up(),
                _ => {}
            },
            UiState::Init => unreachable!(),
            UiState::Quit => {}
        }
    }

    fn render_filter(&mut self, area: Rect, buf: &mut Buffer) {
        self.cursor_position = None;
        if self.state != UiState::Filter {
            return;
        }
        let filter =
            Paragraph::new(self.filter_state.text()).block(Block::bordered().title("Filter"));
        Widget::render(filter, area, buf);
        self.cursor_position = Some(Position::new(
            area.x + self.filter_state.character_index() as u16 + 1,
            area.y + 1,
        ));
    }

    fn render_footer(&mut self, area: Rect, buf: &mut Buffer) {
        match self.state {
            UiState::Browse => {
                Paragraph::new("Use ↓↑ to move, g/G to go top/bottom, forward slash (/) to filter.")
                    .centered()
                    .render(area, buf)
            }
            UiState::Filter => Paragraph::new("Use Esc or Enter to confirm filter.")
                .centered()
                .render(area, buf),
            UiState::Init => unreachable!(),
            UiState::Quit => unreachable!(),
        }
    }

    fn render_header(&self, area: Rect, buf: &mut Buffer) {
        Paragraph::new("safetensors-browser")
            .bold()
            .centered()
            .render(area, buf);
    }

    fn render_list(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::new()
            .title(Line::raw("Tensors").centered())
            .borders(Borders::TOP)
            .border_set(symbols::border::EMPTY)
            .border_style(TODO_HEADER_STYLE)
            .bg(NORMAL_ROW_BG);

        let inner = block.inner(area);
        let [scroll_area, list_area] =
            Layout::horizontal([Constraint::Max(1), Constraint::Fill(1)]).areas(inner); //.areas(block.inner(area));

        let scrollbar = Scrollbar::new(ratatui::widgets::ScrollbarOrientation::VerticalLeft);

        Widget::render(block, area, buf);
        StatefulWidget::render(
            scrollbar,
            scroll_area,
            buf,
            &mut self.tensor_scrollbar_state,
        );

        StatefulWidget::render(&self.tensor_list, list_area, buf, &mut self.tensor_state);
    }

    fn render_selected_item(&mut self, area: Rect, buf: &mut Buffer) {
        let info = if let Some(i) = self.tensor_state.selected() {
            let name = &self.tensor_names[i];
            let metadata = &self.tensors[name];
            let mut info = Vec::new();
            metadata.render_metadata(metadata, &mut info);
            info
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

    fn page_down(&mut self) {
        self.tensor_state.scroll_down_by(10);
        if let Some(position) = self.tensor_state.selected() {
            self.tensor_scrollbar_state = self.tensor_scrollbar_state.position(position);
        }
    }

    fn page_up(&mut self) {
        self.tensor_state.scroll_up_by(10);
        if let Some(position) = self.tensor_state.selected() {
            self.tensor_scrollbar_state = self.tensor_scrollbar_state.position(position);
        }
    }

    fn quit(&mut self) {
        self.state = UiState::Quit;
    }

    fn select_first(&mut self) {
        self.tensor_state.select_first();
        self.tensor_scrollbar_state.first();
    }

    fn select_last(&mut self) {
        self.tensor_state.select_last();
        self.tensor_scrollbar_state.last();
    }

    fn select_next(&mut self) {
        self.tensor_state.select_next();
        self.tensor_scrollbar_state.next();
    }

    fn select_previous(&mut self) {
        self.tensor_state.select_previous();
        self.tensor_scrollbar_state.prev();
    }
}

impl Widget for &mut App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [header_area, main_area, footer_area] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(area);

        let [select_area, detail_area] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Fill(1)]).areas(main_area);

        let [list_area, filter_area] =
            match self.state {
                UiState::Browse => Layout::vertical([Constraint::Fill(1), Constraint::Length(0)])
                    .areas(select_area),
                UiState::Filter => Layout::vertical([Constraint::Fill(1), Constraint::Length(3)])
                    .areas(select_area),
                UiState::Init => unreachable!(),
                UiState::Quit => unreachable!(),
            };

        self.render_header(header_area, buf);
        self.render_list(list_area, buf);
        self.render_filter(filter_area, buf);
        self.render_selected_item(detail_area, buf);
        self.render_footer(footer_area, buf);
    }
}
