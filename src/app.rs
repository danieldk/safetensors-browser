use std::collections::HashMap;

use color_eyre::{owo_colors::colors::xterm::Brown, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Margin, Position, Rect},
    style::{
        palette::{
            material::PINK,
            tailwind::{BLUE, FUCHSIA, SLATE},
        },
        Color, Modifier, Style, Stylize,
    },
    symbols,
    text::{Line, Span},
    widgets::{
        Block, Borders, List, ListState, Padding, Paragraph, ScrollDirection, Scrollbar,
        ScrollbarState, StatefulWidget, Widget, Wrap,
    },
    DefaultTerminal, Frame,
};
use safetensors::tensor::TensorInfo;

use crate::metadata::cmp_numeric_lexicographic;

const TODO_HEADER_STYLE: Style = Style::new().fg(SLATE.c100).bg(BLUE.c800);
const NORMAL_ROW_BG: Color = SLATE.c950;
const ALT_ROW_BG_COLOR: Color = SLATE.c900;
//const SELECTED_STYLE: Style = Style::new().bg(SLATE.c800).add_modifier(Modifier::BOLD);
const SELECTED_STYLE: Style = Style::new().bg(PINK.c600).add_modifier(Modifier::BOLD);
const TEXT_FG_COLOR: Color = SLATE.c200;

#[derive(Debug, Default)]
struct InputState {
    input: String,
    character_index: usize,
}

impl InputState {
    const fn new() -> Self {
        Self {
            input: String::new(),
            character_index: 0,
        }
    }

    fn text(&self) -> &str {
        &self.input
    }

    fn move_cursor_left(&mut self) {
        let cursor_moved_left = self.character_index.saturating_sub(1);
        self.character_index = self.clamp_cursor(cursor_moved_left);
    }

    fn move_cursor_right(&mut self) {
        let cursor_moved_right = self.character_index.saturating_add(1);
        self.character_index = self.clamp_cursor(cursor_moved_right);
    }

    fn enter_char(&mut self, new_char: char) {
        let index = self.byte_index();
        self.input.insert(index, new_char);
        self.move_cursor_right();
    }

    /// Returns the byte index based on the character position.
    ///
    /// Since each character in a string can be contain multiple bytes, it's necessary to calculate
    /// the byte index based on the index of the character.
    fn byte_index(&self) -> usize {
        self.input
            .char_indices()
            .map(|(i, _)| i)
            .nth(self.character_index)
            .unwrap_or(self.input.len())
    }

    fn delete_char(&mut self) {
        let is_not_cursor_leftmost = self.character_index != 0;
        if is_not_cursor_leftmost {
            // Method "remove" is not used on the saved text for deleting the selected char.
            // Reason: Using remove on String works on bytes instead of the chars.
            // Using remove would require special care because of char boundaries.

            let current_index = self.character_index;
            let from_left_to_current_index = current_index - 1;

            // Getting all characters before the selected character.
            let before_char_to_delete = self.input.chars().take(from_left_to_current_index);
            // Getting all characters after selected character.
            let after_char_to_delete = self.input.chars().skip(current_index);

            // Put all characters together except the selected one.
            // By leaving the selected one out, it is forgotten and therefore deleted.
            self.input = before_char_to_delete.chain(after_char_to_delete).collect();
            self.move_cursor_left();
        }
    }

    fn clamp_cursor(&self, new_cursor_pos: usize) -> usize {
        new_cursor_pos.clamp(0, self.input.chars().count())
    }

    fn reset_cursor(&mut self) {
        self.character_index = 0;
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
enum UiState {
    #[default]
    Browse,
    Filter,
    Quit,
}

pub struct App {
    cursor_position: Option<Position>,
    matcher: SkimMatcherV2,
    tensor_names: Vec<String>,
    tensors: HashMap<String, TensorInfo>,
    tensor_state: ListState,
    tensor_scrollbar_state: ScrollbarState,
    state: UiState,
    filter_state: InputState,
}

impl App {
    /// Construct a new instance of [`App`].
    pub fn new(tensors: HashMap<String, TensorInfo>) -> Self {
        let mut tensor_names = tensors.keys().map(ToOwned::to_owned).collect::<Vec<_>>();
        tensor_names.sort_by(|k1, k2| cmp_numeric_lexicographic(k1, k2));
        let scroll_len = tensor_names.len();

        Self {
            cursor_position: None,
            filter_state: Default::default(),
            matcher: Default::default(),
            tensor_names,
            tensors,
            tensor_state: Default::default(),
            tensor_scrollbar_state: ScrollbarState::new(scroll_len),
            state: UiState::Browse,
        }
    }

    /// Run the application's main loop.
    pub fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        while !matches!(self.state, UiState::Quit) {
            // Update tensor list.
            self.tensor_names = self
                .tensors
                .keys()
                .filter(|name| {
                    self.matcher
                        .fuzzy_match(name, self.filter_state.text())
                        .is_some()
                })
                .map(String::clone)
                .collect();

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
                //KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
                //    self.toggle_status();
                // }
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
            area.x + self.filter_state.character_index as u16 + 1,
            area.y + 1,
        ));
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

        let tensors = List::new(self.tensor_names.iter().map(String::as_str))
            .highlight_style(SELECTED_STYLE)
            .highlight_symbol(">")
            .highlight_spacing(ratatui::widgets::HighlightSpacing::Always);

        Widget::render(block, area, buf);
        StatefulWidget::render(
            scrollbar,
            scroll_area,
            buf,
            &mut self.tensor_scrollbar_state,
        );

        StatefulWidget::render(tensors, list_area, buf, &mut self.tensor_state);
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
            Constraint::Length(2),
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
                UiState::Quit => unreachable!(),
            };

        self.render_list(list_area, buf);
        self.render_filter(filter_area, buf);
        self.render_selected_item(detail_area, buf);
    }
}
