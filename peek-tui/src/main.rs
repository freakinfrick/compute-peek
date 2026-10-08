//! peek-tui — the compute-peek `charts` pane, drawn with ratatui.
//!
//! Folding the progress JSONL stays in Python (`compute-peek.py state`), so there is one
//! reader of the contract; this binary only draws what that JSON says, plus the one thing a
//! stateless repaint cannot have: hardware history, kept in memory between ticks. The state
//! is fetched on a background thread so the UI never waits on nvidia-smi.
//!
//! Four tabs. `overview` is the production readout (glance-value order, SPEC §0); `units`,
//! `atlas` and `hardware` are the long-form views and double as a ratatui design showcase:
//! Tabs, BigText, a custom gradient gauge, LineGauge, braille Chart with legend, per-bar
//! styled Sparkline, heat-coloured Table + Scrollbar, BarChart, Canvas scatter, a custom
//! half-block heatmap, and a Clear-backed popup. Every layout reflows to the pane size.
//!
//!     peek-tui --manifest M [--peek compute-peek.py] [--python python3]
//!     peek-tui --manifest M --once [--tab 1..4] [--width 100] [--height 40]  # one frame
//!
//! Keys: 1-4 / Tab / ←→ switch tab · j k ↑↓ scroll units · f follow · r refresh ·
//! ? help · q quit.

use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use ratatui::crossterm::execute;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Circle, Line as CLine, Points};
use ratatui::widgets::{
    Axis, Bar, BarChart, BarGroup, Block, BorderType, Cell, Chart, Clear, Dataset, GraphType,
    LegendPosition, LineGauge, Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Sparkline, SparklineBar, Table, TableState, Tabs, Widget, Wrap,
};
use ratatui::{Frame, Terminal, backend::TestBackend};
use serde_json::Value;
use tui_big_text::{BigText, PixelSize};

const HIST: usize = 300;
const TABS: [&str; 4] = ["overview", "units", "atlas", "hardware"];
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
// ratatui-core/src/symbols/block.rs, left-anchored eighths (index = eighths filled)
const H_EIGHTHS: [&str; 9] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
const PALETTE: [Color; 8] = [
    Color::Cyan,
    Color::Yellow,
    Color::Magenta,
    Color::LightGreen,
    Color::LightBlue,
    Color::LightRed,
    Color::White,
    Color::Green,
];
// viridis, five stops — perceptually even, readable on dark and light terminals
const VIRIDIS: [(u8, u8, u8); 5] =
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)];
const DIM: Color = Color::Rgb(70, 70, 80);

// ================================================================ args + data feed
struct Args {
    manifest: String,
    peek: PathBuf,
    python: String,
    once: bool,
    ansi: bool,
    tab: usize,
    width: u16,
    height: u16,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    // default --peek: <skill>/peek-tui/target/release/peek-tui -> <skill>/compute-peek.py
    let exe = std::env::current_exe()?;
    let skill = exe.ancestors().nth(4).map(PathBuf::from).unwrap_or_default();
    let mut a = Args {
        manifest: String::new(),
        peek: skill.join("compute-peek.py"),
        python: "python3".into(),
        once: false,
        ansi: false,
        tab: 1,
        width: 100,
        height: 40,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--manifest" => a.manifest = val()?,
            "--peek" => a.peek = val()?.into(),
            "--python" => a.python = val()?,
            "--width" => a.width = val()?.parse()?,
            "--height" => a.height = val()?.parse()?,
            "--tab" => a.tab = val()?.parse()?,
            "--once" => a.once = true,
            "--ansi" => a.ansi = true,
            "-h" | "--help" => {
                println!(
                    "peek-tui --manifest M [--peek P] [--python PY] \
                     [--once [--ansi] --tab 1..4 --width W --height H]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {k}").into()),
        }
    }
    if a.manifest.is_empty() {
        return Err("--manifest is required".into());
    }
    Ok(a)
}

fn fetch_state(a: &Args) -> Result<Value, String> {
    let o = Command::new(&a.python)
        .arg(&a.peek)
        .args(["state", "--manifest", &a.manifest])
        .output()
        .map_err(|e| format!("{}: {e}", a.python))?;
    if !o.status.success() {
        return Err(String::from_utf8_lossy(&o.stderr).trim().to_string());
    }
    serde_json::from_slice(&o.stdout).map_err(|e| format!("state JSON: {e}"))
}

/// Poll `state` every `spec.every` seconds on a thread; a send on the returned Sender
/// refreshes now.
fn spawn_fetcher(a: Args) -> (Receiver<Result<Value, String>>, Sender<()>) {
    let (tx, rx) = mpsc::channel();
    let (poke, poked) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        let mut every = Duration::from_secs(1);
        loop {
            let r = fetch_state(&a);
            if let Ok(v) = &r {
                every = Duration::from_secs_f64(v["every"].as_f64().unwrap_or(5.0).max(0.5));
            }
            if tx.send(r).is_err() {
                return;
            }
            if let Err(RecvTimeoutError::Disconnected) = poked.recv_timeout(every) {
                return;
            }
        }
    });
    (rx, poke)
}

// ================================================================ app state
#[derive(Default)]
struct Hist {
    util: VecDeque<f64>,
    temp: VecDeque<f64>,
    power: VecDeque<f64>, // % of limit
}

struct App {
    state: Value,
    err: Option<String>,
    hist: HashMap<String, Hist>,
    updated: Option<Instant>,
    tab: usize,
    help: bool,
    follow: bool,
    table: TableState,
    tick: usize,
}

impl App {
    fn new(tab: usize) -> Self {
        App {
            state: Value::Null,
            err: None,
            hist: HashMap::new(),
            updated: None,
            tab: tab.clamp(1, TABS.len()) - 1,
            help: false,
            follow: true,
            table: TableState::default(),
            tick: 0,
        }
    }

    fn ingest(&mut self, r: Result<Value, String>) {
        match r {
            Ok(v) => {
                self.state = v;
                self.err = None;
                self.updated = Some(Instant::now());
            }
            Err(e) => {
                self.err = Some(e);
                return;
            }
        }
        for g in arr(&self.state["gpus"]) {
            if g.get("error").is_some() {
                continue;
            }
            let h = self.hist.entry(s(&g["device"]).to_string()).or_default();
            push(&mut h.util, f(&g["util"]));
            push(&mut h.temp, f(&g["temp"]));
            let lim = g["power_limit_w"].as_f64().unwrap_or(0.0);
            push(&mut h.power, if lim > 0.0 { f(&g["power_w"]) / lim * 100.0 } else { 0.0 });
        }
    }

    /// Returns false when the app should quit.
    fn key(&mut self, code: KeyCode, mods: KeyModifiers, poke: &Sender<()>) -> bool {
        if self.help && !matches!(code, KeyCode::Char('q')) {
            self.help = false;
            return true;
        }
        let n = arr(&self.state["units"]).len();
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return false,
            KeyCode::Char('c') if mods.contains(KeyModifiers::CONTROL) => return false,
            KeyCode::Char(c @ '1'..='4') => self.tab = c as usize - '1' as usize,
            KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                self.tab = (self.tab + 1) % TABS.len()
            }
            KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => {
                self.tab = (self.tab + TABS.len() - 1) % TABS.len()
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.follow = false;
                let i = self.table.selected().unwrap_or(0);
                self.table.select(Some((i + 1).min(n.saturating_sub(1))));
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.follow = false;
                let i = self.table.selected().unwrap_or(0);
                self.table.select(Some(i.saturating_sub(1)));
            }
            KeyCode::Char('g') => {
                self.follow = false;
                self.table.select(Some(0));
            }
            KeyCode::Char('G') => {
                self.follow = false;
                self.table.select(Some(n.saturating_sub(1)));
            }
            KeyCode::Char('f') => self.follow = true,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('r') => {
                let _ = poke.send(());
            }
            _ => {}
        }
        true
    }

    fn spin(&self) -> &'static str {
        SPINNER[self.tick % SPINNER.len()]
    }

    // ============================================================ frame
    fn render(&mut self, frame: &mut Frame) {
        self.tick += 1;
        let [top, body, foot] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
                .areas(frame.area());
        self.render_tabs(frame, top);
        match self.tab {
            0 => self.overview(frame, body),
            1 => self.units(frame, body),
            2 => self.atlas(frame, body),
            _ => self.hardware(frame, body),
        }
        frame.render_widget(self.footer(), foot);
        if self.help {
            render_help(frame);
        }
    }

    fn render_tabs(&self, frame: &mut Frame, area: Rect) {
        let fresh = match self.updated {
            Some(t) => format!("{} {:.1}s ", self.spin(), t.elapsed().as_secs_f64()),
            None => format!("{} fetching ", self.spin()),
        };
        let [l, r] = Layout::horizontal([
            Constraint::Min(0),
            Constraint::Length(fresh.chars().count() as u16),
        ])
        .areas(area);
        let titles = TABS.iter().enumerate().map(|(i, t)| format!(" {} {t} ", i + 1));
        frame.render_widget(
            Tabs::new(titles)
                .select(self.tab)
                .style(Style::new().fg(Color::Gray))
                .highlight_style(Style::new().fg(Color::Black).bg(Color::Cyan).bold())
                .divider(Span::styled("│", Style::new().fg(DIM)))
                .padding("", ""),
            l,
        );
        frame.render_widget(Paragraph::new(fresh).fg(Color::DarkGray).right_aligned(), r);
    }

    fn footer(&self) -> Paragraph<'_> {
        let st = &self.state;
        let mut spans: Vec<Span> = vec![];
        if let Some(e) = &self.err {
            let last = e.lines().last().unwrap_or("").to_string();
            spans.push(Span::styled(format!(" state error: {last} "), Style::new().red().bold()));
        }
        if b(&st["stale"]) {
            spans.push(Span::styled(" STALE ", Style::new().black().on_red().bold()));
            spans.push(" ".into());
        }
        let hits = st["log"]["hits"].as_u64().unwrap_or(0);
        if hits > 0 {
            spans.push(Span::styled(
                format!("✗ {hits} failure signature(s) "),
                Style::new().red().bold(),
            ));
        } else {
            spans.push(Span::styled("✓ log clean ", Style::new().green()));
        }
        let thought = s(&st["thought"]);
        if !thought.is_empty() {
            spans.push(Span::styled(
                format!("❝ {thought} ❞ "),
                Style::new().magenta().italic(),
            ));
        }
        spans.push(Span::styled("· ? keys", Style::new().fg(Color::DarkGray)));
        Paragraph::new(Line::from(spans))
    }

    // ============================================================ tab 1: overview
    fn overview(&self, frame: &mut Frame, area: Rect) {
        let st = &self.state;
        let groups = arr(&st["groups"]);
        let gpus = arr(&st["gpus"]);
        let hero = area.height >= 38 && area.width >= 96;
        let compact = area.height < 32; // one-line GPUs, two-column groups
        let gcols = if (area.width >= 130 || (compact && area.width >= 70)) && groups.len() > 2 {
            2
        } else {
            1
        };
        let group_h = if groups.is_empty() {
            0
        } else {
            groups.len().div_ceil(gcols).min(8) as u16 + 2
        };
        let tile_h: u16 = if area.height >= 40 {
            6
        } else if area.height >= 28 {
            4
        } else {
            3
        };
        let gpu_stack = !gpus.is_empty() && area.width / (gpus.len() as u16) < 44;
        let gpu_h = match (gpus.len(), gpu_stack) {
            (0, _) => 0,
            (n, _) if compact => n as u16,
            (n, true) => tile_h * n as u16,
            _ => tile_h,
        };
        let [head, hero_a, gauge, grp, charts, hw] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(if hero { 7 } else { 0 }),
            Constraint::Length(3),
            Constraint::Length(group_h),
            Constraint::Min(8),
            Constraint::Length(gpu_h),
        ])
        .areas(area);
        frame.render_widget(self.header(area.width), head);
        if hero {
            self.render_hero(frame, hero_a);
        }
        let frac = f(&st["frac"]).clamp(0.0, 1.0);
        GradientBar {
            ratio: frac,
            label: format!(
                "{}/{} units · {:.0}%",
                st["k"].as_u64().unwrap_or(0),
                st["n"].as_u64().unwrap_or(0),
                frac * 100.0
            ),
            from: (46, 196, 182),
            to: (131, 56, 236),
            block: Some(rounded(format!(" {} ", s(&st["meter_label"])))),
        }
        .render(gauge, frame.buffer_mut());
        self.render_groups(frame, grp, &groups, gcols);
        self.render_chart_grid(frame, charts);
        if compact {
            self.render_gpu_lines(frame, hw, &gpus);
        } else {
            self.render_gpu_strip(frame, hw, &gpus, gpu_stack);
        }
    }

    /// One line per GPU for short panes: name, temp, then a 1-row util history.
    fn render_gpu_lines(&self, frame: &mut Frame, area: Rect, gpus: &[&Value]) {
        let rows = Layout::vertical(vec![Constraint::Length(1); gpus.len()]).split(area);
        for (g, row) in gpus.iter().zip(rows.iter()) {
            let [l, r] = Layout::horizontal([Constraint::Length(30), Constraint::Min(0)]).areas(*row);
            let temp = f(&g["temp"]);
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    format!(" GPU{} {:<12}", g["idx"], truncate(s(&g["name"]), 12)).bold(),
                    format!("{temp:>3.0}°C ").fg(temp_color(temp)),
                    format!("{:>3.0}% ", f(&g["util"])).fg(heat(f(&g["util"]) / 100.0)),
                ])),
                l,
            );
            frame.render_widget(self.util_sparkline(s(&g["device"]), r.width as usize), r);
        }
    }

    /// Util history as per-bar coloured Sparkline, newest at the right edge.
    fn util_sparkline(&self, device: &str, w: usize) -> Sparkline<'_> {
        let util: Vec<f64> =
            self.hist.get(device).map(|h| h.util.iter().copied().collect()).unwrap_or_default();
        let tail = &util[util.len().saturating_sub(w)..];
        let bars: Vec<SparklineBar> = std::iter::repeat_n(SparklineBar::from(None::<u64>), w - tail.len())
            .chain(tail.iter().map(|&u| {
                SparklineBar::from(u as u64).style(Some(Style::new().fg(heat(u / 100.0))))
            }))
            .collect();
        Sparkline::default()
            .data(bars)
            .max(100)
            .absent_value_symbol("·")
            .absent_value_style(Style::new().fg(DIM))
    }

    /// Title yields width first: state + ETA are the glance-value, the title is identity.
    fn header(&self, width: u16) -> Paragraph<'_> {
        let st = &self.state;
        let title = s(&st["title"]);
        let room = (width as usize).saturating_sub(72).max(12);
        let title = if title.chars().count() > room {
            format!("{}…", truncate(title, room - 1))
        } else {
            title.to_string()
        };
        let state = if !b(&st["started"]) {
            Span::styled("waiting for the first heartbeat 💤", Style::new().fg(Color::DarkGray))
        } else if b(&st["ended"]) {
            Span::styled("FINISHED", Style::new().green().bold())
        } else if st["current"].is_string() {
            Span::styled(format!("{} running", self.spin()), Style::new().cyan())
        } else {
            Span::styled("between runs", Style::new().yellow())
        };
        let shards = st["shards"].as_u64().unwrap_or(0);
        Paragraph::new(Line::from(vec![
            Span::styled(format!(" {}  {title}", s(&st["icon"])), Style::new().bold()),
            "   ".into(),
            state,
            if shards > 1 {
                format!(" × {shards} shards").fg(Color::DarkGray)
            } else {
                "".into()
            },
            format!("   elapsed {} · ETA ", fmt_s(st["elapsed_s"].as_f64())).into(),
            Span::styled(fmt_s(st["eta_s"].as_f64()), Style::new().bold()),
            format!(" · done ≈ {}", s(&st["finish"])).into(),
        ]))
    }

    /// Big ETA (tui-big-text, quadrant pixels) beside the stage track and the live unit.
    fn render_hero(&self, frame: &mut Frame, area: Rect) {
        let st = &self.state;
        let eta = if b(&st["ended"]) {
            "done".to_string()
        } else {
            fmt_s(st["eta_s"].as_f64())
        };
        let big_w = (eta.chars().count() as u16) * 4 + 4;
        let [l, r] =
            Layout::horizontal([Constraint::Length(big_w), Constraint::Min(0)]).areas(area);
        let block = rounded(" ETA ".to_string());
        let inner = block.inner(l);
        frame.render_widget(block, l);
        let inner = Rect {
            x: inner.x + 1,
            width: inner.width.saturating_sub(1),
            ..inner
        };
        frame.render_widget(
            BigText::builder()
                .pixel_size(PixelSize::Quadrant)
                .style(Style::new().fg(Color::Cyan).bold())
                .lines(vec![Line::from(eta)])
                .build(),
            inner,
        );

        let block = rounded(" now ".to_string());
        let inner = block.inner(r);
        frame.render_widget(block, r);
        let rows = Layout::vertical([Constraint::Length(1); 5]).split(inner);
        // stage track
        let stages = arr(&st["stages"]);
        let si = st["stage_i"].as_u64().map(|x| x as usize);
        let mut track: Vec<Span> =
            vec![Span::styled(" stage  ", Style::new().fg(Color::DarkGray))];
        for (i, sg) in stages.iter().enumerate() {
            if i > 0 {
                track.push(Span::styled(" ─── ", Style::new().fg(DIM)));
            }
            let style = match si {
                Some(c) if c == i => Style::new().black().on_yellow().bold(),
                Some(c) if i < c => Style::new().fg(Color::Green),
                _ => Style::new().fg(Color::DarkGray),
            };
            track.push(Span::styled(format!(" {} ", s(sg)), style));
        }
        frame.render_widget(Paragraph::new(Line::from(track)), rows[0]);
        // current unit vs its group's typical wall
        let cur = s(&st["current"]).to_string();
        if !cur.is_empty() {
            let now_s = f(&st["now_s"]);
            let typ = st["typ_s"].as_f64();
            let label = match typ {
                Some(t) => format!(
                    " unit   {cur}  {} of ~{}",
                    fmt_s(Some(now_s)),
                    fmt_s(Some(t))
                ),
                None => format!(" unit   {cur}  {}", fmt_s(Some(now_s))),
            };
            frame.render_widget(Paragraph::new(label).bold(), rows[1]);
            if let Some(t) = typ {
                let over = now_s > t * 1.5;
                frame.render_widget(
                    LineGauge::default()
                        .ratio((now_s / t.max(0.001)).min(1.0))
                        .label("        ")
                        .filled_style(Style::new().fg(if over {
                            Color::LightRed
                        } else {
                            Color::Cyan
                        }))
                        .unfilled_style(Style::new().fg(DIM))
                        .filled_symbol("━")
                        .unfilled_symbol("─"),
                    rows[2],
                );
            }
        } else {
            frame.render_widget(Paragraph::new(" unit   —").fg(Color::DarkGray), rows[1]);
        }
        let done = arr(&st["done"]);
        // last finished unit + its first metrics
        if let Some(last) = done.last() {
            let mut spans = vec![
                Span::styled(" last   ", Style::new().fg(Color::DarkGray)),
                Span::styled(format!("{:<14}", s(&last["unit"])), Style::new().bold()),
            ];
            for (i, k) in metric_keys(st).iter().take(4).enumerate() {
                if let Some(v) = last["metrics"][k].as_f64() {
                    spans.push(Span::styled(format!("{k} "), Style::new().fg(Color::DarkGray)));
                    spans.push(Span::styled(
                        format!("{}   ", fmt3(v)),
                        Style::new().fg(PALETTE[(i + 1) % 8]),
                    ));
                }
            }
            frame.render_widget(Paragraph::new(Line::from(spans)), rows[3]);
        }
        let walls: Vec<f64> = done.iter().filter_map(|e| e["wall_s"].as_f64()).collect();
        if !walls.is_empty() {
            let mean = walls.iter().sum::<f64>() / walls.len() as f64;
            frame.render_widget(
                Paragraph::new(format!(
                    " pace   {:.1} s/unit · {:.1} units/min · {} finished",
                    mean,
                    60.0 / mean.max(0.001),
                    walls.len()
                ))
                .fg(Color::Gray),
                rows[4],
            );
        }
    }

    fn render_groups(&self, frame: &mut Frame, area: Rect, groups: &[&Value], gcols: usize) {
        if area.height == 0 {
            return;
        }
        let block = rounded(" groups ".to_string());
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let cells = grid_n(inner, groups.len().min(16), gcols, 2);
        for (i, (g, cell)) in groups.iter().zip(cells).enumerate() {
            let (n, done) = (f(&g["n"]), f(&g["done"]));
            let ratio = if n > 0.0 { (done / n).clamp(0.0, 1.0) } else { 0.0 };
            let col = if done >= n {
                Color::Green
            } else if done > 0.0 {
                PALETTE[i % 8]
            } else {
                Color::DarkGray
            };
            let eta = if done >= n {
                "✓ done".into()
            } else {
                fmt_s(g["eta_s"].as_f64())
            };
            let label = if cell.width >= 50 {
                Line::from(vec![
                    Span::styled(format!("{:>10} ", s(&g["group"])), Style::new().fg(PALETTE[i % 8]).bold()),
                    format!("{:>3}/{:<3} {:<8}", done, n, eta).into(),
                ])
            } else {
                Line::from(vec![
                    Span::styled(format!("{:>6} ", truncate(s(&g["group"]), 6)), Style::new().fg(PALETTE[i % 8]).bold()),
                    format!("{done:>2}/{n:<2}").into(),
                ])
            };
            frame.render_widget(
                LineGauge::default()
                    .ratio(ratio)
                    .label(label)
                    .filled_style(Style::new().fg(col))
                    .unfilled_style(Style::new().fg(DIM))
                    .filled_symbol("━")
                    .unfilled_symbol("─"),
                cell,
            );
        }
    }

    /// Wall time + every metric, one braille chart each; the grid reflows to the pane.
    fn render_chart_grid(&self, frame: &mut Frame, area: Rect) {
        let done = arr(&self.state["done"]);
        if done.is_empty() {
            frame.render_widget(
                Paragraph::new("\n  no unit finished yet — charts appear after the first run_end")
                    .fg(Color::DarkGray)
                    .block(rounded(" charts ".to_string())),
                area,
            );
            return;
        }
        let mut series: Vec<(String, Vec<(f64, f64)>)> =
            vec![("wall s".into(), points(&done, |e| e["wall_s"].as_f64()))];
        for k in metric_keys(&self.state).iter().take(5) {
            series.push((k.clone(), points(&done, |e| e["metrics"][k].as_f64())));
        }
        series.retain(|(_, p)| !p.is_empty());
        // only as many series as fit at ≥ 24 × 7 cells; the rest live in the atlas tab
        let fit = ((area.width as usize / 24) * (area.height as usize / 7)).max(1);
        series.truncate(fit);
        let n = series.len();
        // as many columns as fit at ≥34 cells each; add columns if rows would be < 7 tall
        let mut cols = (area.width as usize / 34).clamp(1, n);
        while cols < n && (area.height as usize) / n.div_ceil(cols) < 7 {
            cols += 1;
        }
        for (i, ((name, pts), cell)) in series.iter().zip(grid_n(area, n, cols, 0)).enumerate() {
            render_series_chart(frame, cell, name, pts, PALETTE[i % 8]);
        }
    }

    fn render_gpu_strip(&self, frame: &mut Frame, area: Rect, gpus: &[&Value], stack: bool) {
        if area.height == 0 || gpus.is_empty() {
            return;
        }
        let cols = if stack { 1 } else { gpus.len() };
        for (g, cell) in gpus.iter().zip(grid_n(area, gpus.len(), cols, 0)) {
            if let Some(e) = g.get("error") {
                frame.render_widget(
                    Paragraph::new(format!("device {} unreadable ({})", s(&g["device"]), s(e)))
                        .fg(Color::DarkGray)
                        .block(rounded(" gpu ".to_string())),
                    cell,
                );
                continue;
            }
            let temp = f(&g["temp"]);
            let title = Line::from(vec![
                format!(" GPU{} {} ", g["idx"], s(&g["name"])).bold(),
                format!("{temp:.0}°C ").fg(temp_color(temp)),
                format!(
                    "· {:.0}% · {:.1}/{:.0} GB · {:.0} W ",
                    f(&g["util"]),
                    f(&g["mem_used_gb"]),
                    f(&g["mem_total_gb"]),
                    f(&g["power_w"])
                )
                .fg(Color::Gray),
            ]);
            let w = cell.width.saturating_sub(2) as usize;
            frame.render_widget(
                self.util_sparkline(s(&g["device"]), w).block(
                    rounded(String::new())
                        .title(title)
                        .title_bottom(Line::from(" util % ").fg(Color::DarkGray)),
                ),
                cell,
            );
        }
    }

    // ============================================================ tab 2: units
    fn units(&mut self, frame: &mut Frame, area: Rect) {
        let wide = area.width >= 110;
        let (t_area, side) = if wide {
            let [a, b] =
                Layout::horizontal([Constraint::Min(60), Constraint::Length(44)]).areas(area);
            (a, b)
        } else {
            let [a, b] =
                Layout::vertical([Constraint::Min(8), Constraint::Length(12)]).areas(area);
            (a, b)
        };
        self.render_unit_table(frame, t_area);
        let [a, b2] = if wide {
            Layout::vertical([Constraint::Percentage(50); 2]).areas(side)
        } else {
            Layout::horizontal([Constraint::Percentage(50); 2]).areas(side)
        };
        self.render_group_bars(frame, a);
        self.render_wall_histogram(frame, b2);
    }

    fn render_unit_table(&mut self, frame: &mut Frame, area: Rect) {
        let st = &self.state;
        let units = arr(&st["units"]);
        let keys = metric_keys(st);
        let done: HashMap<&str, &Value> =
            arr(&st["done"]).into_iter().map(|e| (s(&e["unit"]), e)).collect();
        let live: Vec<&str> = s(&st["current"]).split(" + ").filter(|x| !x.is_empty()).collect();
        // per-metric range, for heat-coloured cells
        let ranges: Vec<(f64, f64)> = keys
            .iter()
            .map(|k| bounds(done.values().filter_map(|e| e["metrics"][k].as_f64())))
            .collect();
        let groups: Vec<String> =
            arr(&st["groups"]).iter().map(|g| s(&g["group"]).to_string()).collect();
        // as many metric columns as fit
        let nk = ((area.width as usize).saturating_sub(2 + 16 + 8 + 8 + 8) / 11).min(keys.len());
        let spin = self.spin();
        let rows: Vec<Row> = units
            .iter()
            .map(|u| {
                let u = s(u);
                let g = unit_group(u);
                let gcol = PALETTE[groups.iter().position(|x| x == g).unwrap_or(0) % 8];
                if let Some(e) = done.get(u) {
                    let mut cells = vec![
                        Cell::from("✓").fg(Color::Green),
                        Cell::from(u.to_string()),
                        Cell::from(g.to_string()).fg(gcol),
                        Cell::from(format!("{:>6.1}s", f(&e["wall_s"]))),
                    ];
                    for (k, (lo, hi)) in keys.iter().zip(&ranges).take(nk) {
                        cells.push(match e["metrics"][k].as_f64() {
                            Some(v) => Cell::from(format!("{:>9}", fmt3(v)))
                                .fg(viridis((v - lo) / (hi - lo))),
                            None => Cell::from(""),
                        });
                    }
                    Row::new(cells)
                } else if live.contains(&u) {
                    Row::new(vec![
                        Cell::from(spin).fg(Color::Cyan),
                        Cell::from(u.to_string()).bold(),
                        Cell::from(g.to_string()).fg(gcol),
                        Cell::from(format!("{:>6.1}s", f(&st["now_s"]))).fg(Color::Cyan),
                        Cell::from("running…").fg(Color::Cyan).italic(),
                    ])
                } else {
                    Row::new(vec![
                        Cell::from("·"),
                        Cell::from(u.to_string()),
                        Cell::from(g.to_string()),
                    ])
                    .fg(Color::DarkGray)
                }
            })
            .collect();
        if self.follow {
            let cur = units
                .iter()
                .position(|u| live.contains(&s(u)))
                .or_else(|| units.iter().position(|u| !done.contains_key(s(u))))
                .unwrap_or(units.len().saturating_sub(1));
            self.table.select(Some(cur));
        }
        let mut widths = vec![
            Constraint::Length(2),
            Constraint::Length(16),
            Constraint::Length(8),
            Constraint::Length(8),
        ];
        widths.extend(std::iter::repeat_n(Constraint::Length(10), nk));
        let mut header: Vec<String> =
            ["", "unit", "group", "  wall"].iter().map(|h| h.to_string()).collect();
        header.extend(keys.iter().take(nk).map(|k| format!("{:>9}", truncate(k, 9))));
        let title = Line::from(vec![
            format!(" units {}/{} ", done.len(), units.len()).bold(),
            if self.follow {
                "following ".fg(Color::Cyan)
            } else {
                "scroll · f follow ".fg(Color::Yellow)
            },
        ]);
        let sel = self.table.selected().unwrap_or(0);
        frame.render_stateful_widget(
            Table::new(rows, widths)
                .header(Row::new(header).fg(Color::Gray).bold())
                .block(rounded(String::new()).title(title))
                .row_highlight_style(Style::new().bg(Color::Rgb(40, 44, 60)))
                .highlight_symbol("▌")
                .column_spacing(1),
            area,
            &mut self.table,
        );
        let mut sb = ScrollbarState::new(units.len()).position(sel);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("╮"))
                .end_symbol(Some("╯"))
                .thumb_style(Style::new().fg(Color::Cyan))
                .track_style(Style::new().fg(DIM)),
            area,
            &mut sb,
        );
    }

    fn render_group_bars(&self, frame: &mut Frame, area: Rect) {
        let groups = arr(&self.state["groups"]);
        let bars: Vec<Bar> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| {
                let m = g["mean_wall"].as_f64()?;
                Some(
                    Bar::default()
                        .value((m * 10.0) as u64)
                        .text_value(format!("{m:.1}"))
                        .label(Line::from(truncate(s(&g["group"]), 6)))
                        .style(Style::new().fg(PALETTE[i % 8]))
                        .value_style(Style::new().fg(Color::Black).bg(PALETTE[i % 8]).bold()),
                )
            })
            .collect();
        let block = rounded(" mean s/unit by group ".to_string());
        if bars.is_empty() {
            frame.render_widget(Paragraph::new(" —").fg(Color::DarkGray).block(block), area);
            return;
        }
        let nb = bars.len() as u16;
        let bw = (area.width.saturating_sub(2 + nb) / nb).clamp(3, 9);
        frame.render_widget(
            BarChart::default()
                .block(block)
                .data(BarGroup::default().bars(&bars))
                .bar_width(bw)
                .bar_gap(1),
            area,
        );
    }

    fn render_wall_histogram(&self, frame: &mut Frame, area: Rect) {
        let walls: Vec<f64> =
            arr(&self.state["done"]).iter().filter_map(|e| e["wall_s"].as_f64()).collect();
        let block = rounded(" wall-time histogram (s) ".to_string());
        if walls.len() < 2 {
            frame.render_widget(Paragraph::new(" —").fg(Color::DarkGray).block(block), area);
            return;
        }
        let (lo, hi) = bounds(walls.iter().copied());
        let inner_w = area.width.saturating_sub(2) as usize;
        let bins = (inner_w / 5).clamp(3, 10);
        let mut counts = vec![0u64; bins];
        for w in &walls {
            let i = (((w - lo) / (hi - lo)) * bins as f64) as usize;
            counts[i.min(bins - 1)] += 1;
        }
        let bars: Vec<Bar> = counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let edge = lo + (hi - lo) * i as f64 / bins as f64;
                Bar::default()
                    .value(c)
                    .label(Line::from(if hi - lo < 10.0 { format!("{edge:.1}") } else { format!("{edge:.0}") }))
                    .style(Style::new().fg(viridis(i as f64 / (bins - 1) as f64)))
            })
            .collect();
        let bw = (inner_w.saturating_sub(bins) / bins).clamp(2, 7) as u16;
        frame.render_widget(
            BarChart::default()
                .block(block)
                .data(BarGroup::default().bars(&bars))
                .bar_width(bw)
                .bar_gap(1)
                .value_style(Style::new().fg(Color::Black).bold()),
            area,
        );
    }

    // ============================================================ tab 3: atlas
    fn atlas(&self, frame: &mut Frame, area: Rect) {
        let n_groups = arr(&self.state["groups"]).len() as u16;
        let heat_h = (n_groups + 4).min(area.height / 2);
        let trend_h = metric_keys(&self.state).len() as u16 + 2;
        if area.width >= 120 {
            let [l, r] = Layout::horizontal([Constraint::Percentage(56), Constraint::Percentage(44)])
                .areas(area);
            let [rt, rm, rb] = Layout::vertical([
                Constraint::Length(heat_h),
                Constraint::Length(trend_h),
                Constraint::Min(0),
            ])
            .areas(r);
            self.render_scatter(frame, l);
            self.render_heatmap(frame, rt);
            self.render_trend_bars(frame, rm);
            self.render_group_profile(frame, rb);
        } else {
            let [t, m, bt] = Layout::vertical([
                Constraint::Min(10),
                Constraint::Length(heat_h),
                Constraint::Length(trend_h),
            ])
            .areas(area);
            self.render_scatter(frame, t);
            self.render_heatmap(frame, m);
            self.render_trend_bars(frame, bt);
        }
    }

    /// metric[0] vs metric[1], coloured by group, with a trail through the last units.
    fn render_scatter(&self, frame: &mut Frame, area: Rect) {
        let st = &self.state;
        let keys = metric_keys(st);
        let (Some(kx), Some(ky)) = (keys.first(), keys.get(1)) else {
            frame.render_widget(
                Paragraph::new(" needs two metrics").block(rounded(" phase portrait ".into())),
                area,
            );
            return;
        };
        let pts: Vec<(String, f64, f64)> = arr(&st["done"])
            .iter()
            .filter_map(|e| {
                Some((
                    s(&e["group"]).to_string(),
                    e["metrics"][kx].as_f64()?,
                    e["metrics"][ky].as_f64()?,
                ))
            })
            .collect();
        let groups: Vec<String> =
            arr(&st["groups"]).iter().map(|g| s(&g["group"]).to_string()).collect();
        let pad = |(a, b): (f64, f64)| (a - (b - a) * 0.06, b + (b - a) * 0.06);
        let (x0, x1) = pad(bounds(pts.iter().map(|p| p.1)));
        let (y0, y1) = pad(bounds(pts.iter().map(|p| p.2)));
        // quartile crosshair ticks, not full rules: structure without noise
        let (dx, dy) = ((x1 - x0) / 60.0, (y1 - y0) / 30.0);
        let grid: Vec<(f64, f64)> = (1..4)
            .flat_map(|i| (1..4).map(move |j| (i, j)))
            .flat_map(|(i, j)| {
                let (gx, gy) = (x0 + (x1 - x0) * i as f64 / 4.0, y0 + (y1 - y0) * j as f64 / 4.0);
                (-2..=2).flat_map(move |k| [(gx + dx * k as f64, gy), (gx, gy + dy * k as f64)])
            })
            .collect();
        let by_group: Vec<(Color, Vec<(f64, f64)>)> = groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                (PALETTE[i % 8], pts.iter().filter(|p| &p.0 == g).map(|p| (p.1, p.2)).collect())
            })
            .collect();
        let trail: Vec<(f64, f64)> = pts.iter().rev().take(10).map(|p| (p.1, p.2)).collect();
        let r = (x1 - x0) * 0.025;
        let legend: Vec<Span> = groups
            .iter()
            .enumerate()
            .flat_map(|(i, g)| {
                [Span::styled(" ● ", Style::new().fg(PALETTE[i % 8])), Span::raw(g.clone())]
            })
            .chain([Span::styled("  ◯ latest ", Style::new().fg(Color::Yellow))])
            .collect();
        let title = Line::from(vec![
            " phase portrait ".bold(),
            format!("{kx} → · {ky} ↑ ").fg(Color::Gray),
        ]);
        frame.render_widget(
            Canvas::default()
                .block(rounded(String::new()).title(title).title_bottom(Line::from(legend)))
                .marker(Marker::Braille)
                .x_bounds([x0, x1])
                .y_bounds([y0, y1])
                .paint(|ctx| {
                    ctx.draw(&Points { coords: &grid, color: DIM });
                    ctx.layer();
                    for w in trail.windows(2) {
                        ctx.draw(&CLine::new(w[0].0, w[0].1, w[1].0, w[1].1, Color::Gray));
                    }
                    ctx.layer();
                    for (c, p) in &by_group {
                        ctx.draw(&Points { coords: p, color: *c });
                    }
                    if let Some(&(x, y)) = trail.first() {
                        ctx.draw(&Circle { x, y, radius: r, color: Color::Yellow });
                    }
                    ctx.print(x0, y1, Line::from(fmt3(y1)).fg(Color::DarkGray));
                    ctx.print(x0, y0, Line::from(fmt3(y0)).fg(Color::DarkGray));
                }),
            area,
        );
    }

    /// groups × units grid, each cell a half-block pair shaded by one metric (viridis).
    fn render_heatmap(&self, frame: &mut Frame, area: Rect) {
        let st = &self.state;
        let keys = metric_keys(st);
        let key = keys.get(2).or(keys.first()).cloned().unwrap_or_default();
        let block = rounded(String::new()).title(Line::from(vec![
            " unit map ".bold(),
            format!("shaded by {key} ").fg(Color::Gray),
        ]));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let units = arr(&st["units"]);
        let done: HashMap<&str, &Value> =
            arr(&st["done"]).into_iter().map(|e| (s(&e["unit"]), e)).collect();
        let live: Vec<&str> = s(&st["current"]).split(" + ").collect();
        let (lo, hi) = bounds(done.values().filter_map(|e| e["metrics"][&key].as_f64()));
        let groups: Vec<String> =
            arr(&st["groups"]).iter().map(|g| s(&g["group"]).to_string()).collect();
        let blink = self.tick % 4 < 2;
        let buf = frame.buffer_mut();
        for (gi, g) in groups.iter().enumerate() {
            let y = inner.y + gi as u16;
            if y + 2 > inner.bottom() {
                break;
            }
            buf.set_string(
                inner.x,
                y,
                format!("{:>8} ", truncate(g, 8)),
                Style::new().fg(PALETTE[gi % 8]),
            );
            let row = units.iter().map(|u| s(u)).filter(|u| unit_group(u) == g.as_str());
            for (ci, u) in row.enumerate() {
                let x = inner.x + 9 + ci as u16 * 3;
                if x + 2 > inner.right() {
                    break;
                }
                let (sym, style) = if let Some(e) = done.get(u) {
                    match e["metrics"][&key].as_f64() {
                        Some(v) => ("██", Style::new().fg(viridis((v - lo) / (hi - lo)))),
                        None => ("▒▒", Style::new().fg(Color::Gray)),
                    }
                } else if live.contains(&u) {
                    (if blink { "▐▌" } else { "  " }, Style::new().fg(Color::Cyan).bold())
                } else {
                    ("··", Style::new().fg(DIM))
                };
                buf.set_string(x, y, sym, style);
            }
        }
        // colour bar: two viridis steps per cell via an upper half block
        let y = inner.bottom().saturating_sub(1);
        if inner.height >= 3 && inner.width > 24 {
            let w = (inner.width - 22).min(40);
            buf.set_string(inner.x, y, format!("{:>8} ", fmt3(lo)), Style::new().fg(Color::DarkGray));
            for i in 0..w {
                let t = |k: f64| viridis((i as f64 * 2.0 + k) / (w as f64 * 2.0 - 1.0));
                buf.set_string(inner.x + 9 + i, y, "▌", Style::new().fg(t(0.0)).bg(t(1.0)));
            }
            buf.set_string(inner.x + 10 + w, y, fmt3(hi), Style::new().fg(Color::DarkGray));
        }
    }

    /// Grouped BarChart: each group's mean of every metric, normalised to the run's range.
    fn render_group_profile(&self, frame: &mut Frame, area: Rect) {
        if area.height < 6 {
            return;
        }
        let st = &self.state;
        let keys = metric_keys(st);
        let done = arr(&st["done"]);
        let ranges: Vec<(f64, f64)> = keys
            .iter()
            .map(|k| bounds(done.iter().filter_map(|e| e["metrics"][k].as_f64())))
            .collect();
        let legend: Vec<Span> = keys
            .iter()
            .enumerate()
            .flat_map(|(i, k)| [Span::styled(" ■ ", Style::new().fg(PALETTE[(i + 1) % 8])), Span::raw(k.clone())])
            .collect();
        let mut chart = BarChart::default()
            .block(
                rounded(" group profile · metric means, % of run range ".to_string())
                    .title_bottom(Line::from(legend)),
            )
            .bar_width(2)
            .bar_gap(0)
            .group_gap(3)
            .max(100);
        for g in arr(&st["groups"]) {
            let gname = s(&g["group"]);
            let mine: Vec<&&Value> = done.iter().filter(|e| s(&e["group"]) == gname).collect();
            if mine.is_empty() {
                continue;
            }
            let bars: Vec<Bar> = keys
                .iter()
                .zip(&ranges)
                .enumerate()
                .map(|(i, (k, (lo, hi)))| {
                    let vs: Vec<f64> = mine.iter().filter_map(|e| e["metrics"][k].as_f64()).collect();
                    let m = vs.iter().sum::<f64>() / vs.len().max(1) as f64;
                    Bar::default()
                        .value((((m - lo) / (hi - lo)).clamp(0.0, 1.0) * 100.0) as u64)
                        .text_value(String::new())
                        .style(Style::new().fg(PALETTE[(i + 1) % 8]))
                })
                .collect();
            chart = chart.data(BarGroup::default().label(Line::from(gname.to_string()).centered()).bars(&bars));
        }
        frame.render_widget(chart, area);
    }

    /// One LineGauge per metric: the last value within its min..max so far, with trend.
    fn render_trend_bars(&self, frame: &mut Frame, area: Rect) {
        let st = &self.state;
        let done = arr(&st["done"]);
        let block = rounded(" where each metric sits · last within range ".to_string());
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let keys = metric_keys(st);
        let cells = grid_n(inner, keys.len().min(inner.height as usize), 1, 0);
        for (i, (k, cell)) in keys.iter().zip(cells).enumerate() {
            let vs: Vec<f64> = done.iter().filter_map(|e| e["metrics"][k].as_f64()).collect();
            let Some(&last) = vs.last() else { continue };
            let (lo, hi) = bounds(vs.iter().copied());
            let prev = if vs.len() > 5 {
                vs[vs.len() - 6..vs.len() - 1].iter().sum::<f64>() / 5.0
            } else {
                last
            };
            let arrow = trend_arrow(prev, last, hi - lo);
            let c = PALETTE[(i + 1) % 8];
            frame.render_widget(
                LineGauge::default()
                    .ratio(((last - lo) / (hi - lo)).clamp(0.0, 1.0))
                    .label(Line::from(vec![
                        Span::styled(format!("{:>12} ", truncate(k, 12)), Style::new().fg(c).bold()),
                        Span::styled(format!("{arrow} {:<8}", fmt3(last)), Style::new().fg(c)),
                    ]))
                    .filled_style(Style::new().fg(c))
                    .unfilled_style(Style::new().fg(DIM))
                    .filled_symbol("█")
                    .unfilled_symbol("░"),
                cell,
            );
        }
    }

    // ============================================================ tab 4: hardware
    fn hardware(&self, frame: &mut Frame, area: Rect) {
        let gpus = arr(&self.state["gpus"]);
        if gpus.is_empty() {
            frame.render_widget(
                Paragraph::new("\n  no \"gpu\" in the spec — add device UUIDs (compute-peek.py gpus)")
                    .fg(Color::DarkGray)
                    .block(rounded(" hardware ".into())),
                area,
            );
            return;
        }
        let cols = if area.width as usize >= 50 * gpus.len() {
            gpus.len()
        } else if area.width >= 112 {
            2
        } else {
            1
        };
        for (g, cell) in gpus.iter().zip(grid_n(area, gpus.len(), cols, 0)) {
            self.render_gpu_card(frame, cell, g);
        }
    }

    fn render_gpu_card(&self, frame: &mut Frame, area: Rect, g: &Value) {
        let block = Block::bordered()
            .border_type(BorderType::Double)
            .border_style(Style::new().fg(Color::DarkGray))
            .title(Line::from(format!(" GPU{} · {} ", g["idx"], s(&g["name"]))).bold().cyan())
            .title_bottom(
                Line::from(format!(" {} ", truncate(s(&g["device"]), 20)))
                    .fg(Color::DarkGray)
                    .right_aligned(),
            );
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if let Some(e) = g.get("error") {
            frame.render_widget(Paragraph::new(format!("unreadable ({})", s(e))).fg(Color::Red), inner);
            return;
        }
        let [u, m, p, t, _, ch] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .areas(inner);
        let util = f(&g["util"]);
        GradientBar {
            ratio: util / 100.0,
            label: format!("util {util:.0}%"),
            from: (46, 196, 182),
            to: (255, 89, 94),
            block: None,
        }
        .render(u, frame.buffer_mut());
        let (mu, mt) = (f(&g["mem_used_gb"]), f(&g["mem_total_gb"]).max(0.001));
        frame.render_widget(meter(format!("vram {mu:>5.1}/{mt:.0} GB"), mu / mt, Color::LightBlue), m);
        let (pw, pl) = (f(&g["power_w"]), g["power_limit_w"].as_f64());
        let plabel = match pl {
            Some(l) => format!("power {pw:>4.0}/{l:.0} W"),
            None => format!("power {pw:>4.0} W"),
        };
        frame.render_widget(meter(plabel, pl.map_or(0.0, |l| pw / l.max(1.0)), Color::Yellow), p);
        let temp = f(&g["temp"]);
        frame.render_widget(meter(format!("temp  {temp:>4.0} °C"), temp / 100.0, temp_color(temp)), t);

        let Some(h) = self.hist.get(s(&g["device"])) else { return };
        if ch.height < 5 {
            return;
        }
        let to_pts = |q: &VecDeque<f64>| -> Vec<(f64, f64)> {
            q.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect()
        };
        let (pu, pt, pp) = (to_pts(&h.util), to_pts(&h.temp), to_pts(&h.power));
        let xmax = (h.util.len().max(2) - 1) as f64;
        let wide = ch.width >= 40;
        frame.render_widget(
            Chart::new(vec![
                Dataset::default().name("util %").marker(Marker::Braille)
                    .graph_type(GraphType::Line).style(Style::new().fg(Color::Cyan)).data(&pu),
                Dataset::default().name("temp °C").marker(Marker::Braille)
                    .graph_type(GraphType::Line).style(Style::new().fg(Color::LightRed)).data(&pt),
                Dataset::default().name("power %").marker(Marker::Braille)
                    .graph_type(GraphType::Line).style(Style::new().fg(Color::Yellow)).data(&pp),
            ])
            .block(Block::new().title(
                Line::from(format!(" history · {} samples ", h.util.len())).fg(Color::DarkGray),
            ))
            .legend_position(if wide { Some(LegendPosition::TopLeft) } else { None })
            .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)))
            .x_axis(Axis::default().bounds([0.0, xmax]).style(Style::new().fg(DIM)))
            .y_axis(
                Axis::default()
                    .bounds([0.0, 100.0])
                    .labels(if wide { vec!["0", "50", "100"] } else { vec![] })
                    .style(Style::new().fg(Color::DarkGray)),
            ),
            ch,
        );
    }
}

// ================================================================ custom widgets
/// A gauge filled in 1/8-cell steps along an RGB gradient, label inverted over the fill.
struct GradientBar<'a> {
    ratio: f64,
    label: String,
    from: (u8, u8, u8),
    to: (u8, u8, u8),
    block: Option<Block<'a>>,
}

impl Widget for GradientBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let inner = match self.block {
            Some(b) => {
                let i = b.inner(area);
                b.render(area, buf);
                i
            }
            None => area,
        };
        if inner.is_empty() {
            return;
        }
        let w = inner.width as f64;
        let filled = self.ratio.clamp(0.0, 1.0) * w;
        let label: Vec<char> = self.label.chars().collect();
        let ly = inner.y + inner.height / 2;
        let lx = inner.x + inner.width.saturating_sub(label.len() as u16) / 2;
        for y in inner.top()..inner.bottom() {
            for i in 0..inner.width {
                let x = inner.x + i;
                let c = lerp(self.from, self.to, i as f64 / (w - 1.0).max(1.0));
                let cf = i as f64;
                let full = cf + 1.0 <= filled;
                let cell = &mut buf[(x, y)];
                if y == ly && x >= lx && ((x - lx) as usize) < label.len() {
                    let ch = label[(x - lx) as usize].to_string();
                    let style = if full {
                        Style::new().fg(Color::Black).bg(c).bold()
                    } else {
                        Style::new().fg(Color::White).bold()
                    };
                    cell.set_symbol(&ch).set_style(style);
                } else if full {
                    cell.set_symbol("█").set_fg(c);
                } else if cf < filled {
                    let e = (((filled - cf) * 8.0).round() as usize).clamp(1, 8);
                    cell.set_symbol(H_EIGHTHS[e]).set_fg(c);
                } else {
                    cell.set_symbol("░").set_fg(DIM);
                }
            }
        }
    }
}

fn render_series_chart(frame: &mut Frame, area: Rect, name: &str, pts: &[(f64, f64)], color: Color) {
    let (lo, hi) = bounds(pts.iter().map(|p| p.1));
    let pad = (hi - lo) * 0.08;
    let xmax = pts.len().max(2) as f64;
    let last = pts.last().map_or(0.0, |p| p.1);
    // rolling mean over 5: the trend under the noise
    let mean: Vec<(f64, f64)> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let w = &pts[i.saturating_sub(4)..=i];
            (p.0, w.iter().map(|q| q.1).sum::<f64>() / w.len() as f64)
        })
        .collect();
    let prev = mean.iter().rev().nth(3).map_or(last, |p| p.1);
    let cur = mean.last().map_or(last, |p| p.1);
    let roomy = area.width >= 30 && area.height >= 9;
    let title = Line::from(vec![
        format!(" {name} ").fg(color).bold(),
        format!("{} {} ", trend_arrow(prev, cur, hi - lo), fmt3(last)).fg(color),
    ]);
    let mut xa = Axis::default().bounds([1.0, xmax]).style(Style::new().fg(DIM));
    let mut ya = Axis::default().bounds([lo - pad, hi + pad]).style(Style::new().fg(Color::DarkGray));
    if roomy {
        xa = xa.labels(["1".to_string(), format!("{}", pts.len())]);
        ya = ya.labels([fmt3(lo), fmt3(hi)]);
    }
    frame.render_widget(
        Chart::new(vec![
            Dataset::default()
                .name("raw")
                .marker(Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(Style::new().fg(color))
                .data(pts),
            Dataset::default()
                .name("mean₅")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::new().fg(Color::Gray))
                .data(&mean),
        ])
        .block(rounded(String::new()).title(title))
        .legend_position(if area.height >= 14 { Some(LegendPosition::TopRight) } else { None })
        .hidden_legend_constraints((Constraint::Ratio(1, 3), Constraint::Ratio(1, 3)))
        .x_axis(xa)
        .y_axis(ya),
        area,
    );
}

fn render_help(frame: &mut Frame) {
    let [a] = Layout::horizontal([Constraint::Length(54)]).flex(Flex::Center).areas(frame.area());
    let [a] = Layout::vertical([Constraint::Length(15)]).flex(Flex::Center).areas(a);
    frame.render_widget(Clear, a);
    let key = |k: &str, d: &str| {
        Line::from(vec![format!("  {k:<14}").cyan().bold(), d.to_string().into()])
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            key("1 2 3 4", "overview · units · atlas · hardware"),
            key("tab ← →  h l", "cycle tabs"),
            key("j k ↑ ↓", "scroll the unit table"),
            key("g G", "top / bottom of the table"),
            key("f", "follow the running unit again"),
            key("r", "refresh now"),
            key("?", "this card (any key closes)"),
            key("q esc", "quit"),
            Line::from(""),
            Line::from("  data: compute-peek.py state · repaint 4 Hz".fg(Color::DarkGray)),
            Line::from("  widgets: ratatui 0.30 + tui-big-text".fg(Color::DarkGray)),
        ])
        .wrap(Wrap { trim: false })
        .block(
            Block::bordered()
                .border_type(BorderType::Thick)
                .border_style(Style::new().fg(Color::Cyan))
                .title(Line::from(" keys ").bold().centered()),
        ),
        a,
    );
}

// ================================================================ small helpers
fn rounded<'a>(title: String) -> Block<'a> {
    let b = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::DarkGray));
    if title.is_empty() {
        b
    } else {
        b.title(Line::from(title).fg(Color::Gray))
    }
}

fn meter<'a>(label: String, ratio: f64, color: Color) -> LineGauge<'a> {
    LineGauge::default()
        .ratio(ratio.clamp(0.0, 1.0))
        .label(Span::styled(format!("{label:<22}"), Style::new().fg(Color::Gray)))
        .filled_style(Style::new().fg(color))
        .unfilled_style(Style::new().fg(DIM))
        .filled_symbol("━")
        .unfilled_symbol("─")
}

/// `n` cells in at most `cols` columns, balanced (5 in ≤4 → 3 + 2) with rows sharing the
/// height and the last row stretched; `gap` cells between columns.
fn grid_n(area: Rect, n: usize, cols: usize, gap: u16) -> Vec<Rect> {
    if n == 0 {
        return vec![];
    }
    let rows = n.div_ceil(cols.clamp(1, n));
    let cols = n.div_ceil(rows);
    let mut out = Vec::with_capacity(n);
    for (r, rr) in Layout::vertical(vec![Constraint::Ratio(1, rows as u32); rows])
        .split(area)
        .iter()
        .enumerate()
    {
        let in_row = (n - r * cols).min(cols);
        out.extend(
            Layout::horizontal(vec![Constraint::Ratio(1, in_row as u32); in_row])
                .spacing(gap)
                .split(*rr)
                .iter()
                .copied(),
        );
    }
    out
}

fn metric_keys(st: &Value) -> Vec<String> {
    let keys: Vec<String> = arr(&st["metrics"]).iter().map(|k| s(k).to_string()).collect();
    if !keys.is_empty() {
        return keys;
    }
    match arr(&st["done"]).last().map(|e| &e["metrics"]) {
        Some(Value::Object(m)) => m.keys().cloned().collect(),
        _ => vec![],
    }
}

/// Same rule as compute-peek.py `unit_group`.
fn unit_group(u: &str) -> &str {
    match u.split_once('=') {
        Some((g, _)) => g,
        None => u.split_whitespace().next().unwrap_or(""),
    }
}

fn trend_arrow(prev: f64, cur: f64, span: f64) -> &'static str {
    if cur > prev + span * 0.03 {
        "▲"
    } else if cur < prev - span * 0.03 {
        "▼"
    } else {
        "■"
    }
}

fn arr(v: &Value) -> Vec<&Value> {
    v.as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn b(v: &Value) -> bool {
    v.as_bool().unwrap_or(false)
}

fn truncate(t: &str, n: usize) -> String {
    t.chars().take(n).collect()
}

fn push(q: &mut VecDeque<f64>, v: f64) {
    if q.len() == HIST {
        q.pop_front();
    }
    q.push_back(v);
}

fn points(done: &[&Value], get: impl Fn(&Value) -> Option<f64>) -> Vec<(f64, f64)> {
    done.iter()
        .enumerate()
        .filter_map(|(i, e)| get(e).map(|v| ((i + 1) as f64, v)))
        .collect()
}

/// min..max of the values, widened when degenerate so every divide by (hi - lo) is safe.
fn bounds(vs: impl Iterator<Item = f64>) -> (f64, f64) {
    let (lo, hi) = vs.fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(v), b.max(v)));
    if lo > hi {
        (0.0, 1.0)
    } else if hi - lo < 1e-9 {
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    }
}

fn lerp(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
    Color::Rgb(m(a.0, b.0), m(a.1, b.1), m(a.2, b.2))
}

fn viridis(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0) * (VIRIDIS.len() - 1) as f64;
    let i = (t.floor() as usize).min(VIRIDIS.len() - 2);
    lerp(VIRIDIS[i], VIRIDIS[i + 1], t - i as f64)
}

/// green → amber → red, for load.
fn heat(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        lerp((80, 200, 120), (255, 191, 0), t * 2.0)
    } else {
        lerp((255, 191, 0), (255, 70, 70), (t - 0.5) * 2.0)
    }
}

fn temp_color(t: f64) -> Color {
    if t >= 80.0 {
        Color::Red
    } else if t >= 70.0 {
        Color::Yellow
    } else {
        Color::Green
    }
}

/// Three significant digits, like the Python readout's `:.3g`.
fn fmt3(v: f64) -> String {
    let a = v.abs();
    if a >= 1000.0 || (a > 0.0 && a < 0.001) {
        format!("{v:.2e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

/// Same shape as compute-peek.py `fmt_s`.
fn fmt_s(v: Option<f64>) -> String {
    let Some(v) = v else { return "—".into() };
    if v < 10.0 && v.fract() != 0.0 {
        return format!("{:.1}s", v.max(0.0));
    }
    let s = v.max(0.0) as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else if s < 86400 {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    } else {
        format!("{}d {:02}h", s / 86400, (s % 86400) / 3600)
    }
}

/// One buffer row as SGR-coloured text (`--once --ansi`, for screenshots and docs).
fn ansi_line(buf: &Buffer, y: u16, w: u16) -> String {
    fn sgr(c: Color, fg: bool) -> String {
        let base = if fg { 30 } else { 40 };
        match c {
            Color::Reset => format!("{}", base + 9),
            Color::Rgb(r, g, b) => format!("{};2;{r};{g};{b}", base + 8),
            Color::Indexed(i) => format!("{};5;{i}", base + 8),
            named => {
                let order = [
                    Color::Black, Color::Red, Color::Green, Color::Yellow, Color::Blue,
                    Color::Magenta, Color::Cyan, Color::Gray, Color::DarkGray,
                    Color::LightRed, Color::LightGreen, Color::LightYellow, Color::LightBlue,
                    Color::LightMagenta, Color::LightCyan, Color::White,
                ];
                let i = order.iter().position(|o| *o == named).unwrap_or(7) as u16;
                format!("{}", if i < 8 { base + i } else { base + 60 + i - 8 })
            }
        }
    }
    let mut out = String::new();
    let mut last: Option<(Color, Color, Modifier)> = None;
    for x in 0..w {
        let c = &buf[(x, y)];
        let st = (c.fg, c.bg, c.modifier);
        if last != Some(st) {
            out.push_str(&format!("\x1b[0;{};{}", sgr(c.fg, true), sgr(c.bg, false)));
            if c.modifier.contains(Modifier::BOLD) {
                out.push_str(";1");
            }
            if c.modifier.contains(Modifier::DIM) {
                out.push_str(";2");
            }
            out.push('m');
            last = Some(st);
        }
        out.push_str(c.symbol());
    }
    out + "\x1b[0m"
}

// ================================================================ main
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let mut app = App::new(args.tab);

    if args.once {
        app.ingest(fetch_state(&args));
        let (w, h) = (args.width, args.height);
        let mut term = Terminal::new(TestBackend::new(w, h))?;
        term.draw(|fr| app.render(fr))?;
        let buf = term.backend().buffer();
        for y in 0..h {
            if args.ansi {
                println!("{}", ansi_line(buf, y, w));
                continue;
            }
            let line: String = (0..w).map(|x| buf[(x, y)].symbol()).collect();
            println!("{}", line.trim_end());
        }
        return Ok(());
    }

    let (rx, poke) = spawn_fetcher(args);
    let mut term = ratatui::init();
    // pasted text (e.g. a mux's send-keys/send-text) arrives as one Paste event, not key presses
    execute!(std::io::stdout(), EnableBracketedPaste)?;
    let res = (|| -> Result<(), Box<dyn Error>> {
        loop {
            while let Ok(r) = rx.try_recv() {
                app.ingest(r);
            }
            term.draw(|fr| app.render(fr))?;
            if !event::poll(Duration::from_millis(250))? {
                continue;
            }
            let keep = match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    app.key(k.code, k.modifiers, &poke)
                }
                Event::Paste(text) => text
                    .chars()
                    .all(|c| app.key(KeyCode::Char(c), KeyModifiers::NONE, &poke)),
                _ => true,
            };
            if !keep {
                return Ok(());
            }
        }
    })();
    let _ = execute!(std::io::stdout(), DisableBracketedPaste);
    ratatui::restore();
    res
}
