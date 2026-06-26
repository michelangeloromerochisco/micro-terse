"""Terminal CLI for Micro-Terse.

Usage:
    python scripts/chat.py
    python scripts/chat.py --url http://127.0.0.1:8080

A Claude Code-inspired terminal UI for Micro-Terse, built with Rich.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from typing import Any

import httpx

try:
    from rich import box
    from rich.console import Console, RenderableType
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.status import Status
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except Exception as _rich_err:  # pragma: no cover - tested via optional dep path
    _HAS_RICH = False
    _RICH_ERROR = str(_rich_err)


def _configure_windows_console() -> None:
    """Force UTF-8 input/output codepages on Windows for box-drawing glyphs."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Theme tokens (warm yellow / Claude Code inspired)
# ---------------------------------------------------------------------------
_COLOR_YELLOW = "#FACC15"      # bright amber/yellow for primary accents
_COLOR_GOLD = "#EAB308"        # deeper gold for borders and headings
_COLOR_USER = "#FDE047"        # pale yellow for user
_COLOR_DIM = "#A1A1AA"         # muted gray for secondary text
_COLOR_MUTED = "#71717A"       # dimmer gray for tertiary text
_COLOR_ERROR = "#EF4444"       # red for errors
_COLOR_ONLINE = "#22C55E"      # green online dot
_COLOR_OFFLINE = "#EF4444"     # red offline dot

_SPINNER = "dots"
_AGENT_GLYPH = "✻"
_USER_GLYPH = "▸"


# ---------------------------------------------------------------------------
# Demo content: spec sheet + curated starter prompts
# ---------------------------------------------------------------------------
# The chat backend is the SFT checkpoint — the most fluent one, but it has no
# identity (that lives in the ORPO model, shown via /proof). So the starters are
# open-ended generation prompts that play to a small model's strength, not
# identity questions (which would answer "ChatGPT" here and contradict /proof).
_SUGGESTED_PROMPTS = [
    "Tell me about a small town in France.",
    "Describe a sunny morning by the sea.",
    "Write a few sentences about the ocean.",
    "Tell me a short story about a traveler.",
    "Why is reading good for you?",
]

# (label, value) pairs for the /about spec card. Kept honest: sizes are the
# real F32 GGUF and the projected TQ2_0 quantization.
_SPEC_FACTS = [
    ("Model", "Terse-Micro · 423M parameters"),
    ("Weights", "ternary {-1, 0, +1} — ~1.6 bits/weight"),
    ("Trained", "from scratch: 8B tokens + chat SFT + identity ORPO"),
    ("Size", "1.7 GB (F32 GGUF) → ~182 MB quantized (TQ2_0)"),
    ("Runtime", "CPU-only — no GPU required"),
    ("Stack", "100% custom PyTorch: MoE · MTP · QK-norm · ternary"),
    ("Identity", "first Colombian ternary-weight LLM"),
    ("Author", "Michelangelo Romero Chisco"),
]


# ---------------------------------------------------------------------------
# Logo (kept as plain strings so it renders in any fixed-width terminal)
# ---------------------------------------------------------------------------
_LOGO_LEFT = [
    "╭───────╮",
    "│       │",
    "│  t ▌  │",
    "│       │",
    "╰───────╯",
]
_LOGO_RIGHT = [
    "████████ ███████ ██████  ███████ ███████",
    "   ██    ██      ██   ██ ██      ██     ",
    "   ██    █████   ██████  ███████ █████  ",
    "   ██    ██      ██   ██      ██ ██     ",
    "   ██    ███████ ██   ██ ███████ ███████",
]


# Position of the animated block cursor inside the logo box.
_LOGO_CURSOR_ROW = 3       # 1-indexed terminal row of the "│  t ▌  │" line after the header is printed
_LOGO_CURSOR_COL = 6       # 1-indexed terminal column where the ▌/space sits


def _ansi_fg(hex_color: str) -> str:
    """Convert a hex color like #EAB308 to a 24-bit ANSI foreground escape."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


def _format_logo(block_on: bool = True) -> Text:
    """Render the Micro-Terse ASCII logo in warm yellow.

    The block cursor inside the box (t ▌) toggles with ``block_on``.
    """
    marker = "▌" if block_on else " "
    left_lines = [
        "╭───────╮",
        "│       │",
        f"│  t {marker}  │",
        "│       │",
        "╰───────╯",
    ]
    text = Text()
    for left, right in zip(left_lines, _LOGO_RIGHT):
        text.append(left, style=f"bold {_COLOR_GOLD}")
        text.append("  ", style="default")
        text.append(right, style=f"bold {_COLOR_YELLOW}")
        text.append("\n")
    return text


# Static logo used by callers/tests that do not need the animated cursor.
_FORMATTED_STATIC_LOGO = _format_logo(block_on=True)


# ---------------------------------------------------------------------------
# Rendering helpers (pure, testable)
# ---------------------------------------------------------------------------
def _parse_thinking(text: str) -> tuple[str, str]:
    """Split content into reasoning and main answer.

    Returns (thinking, main). Multiple thinking blocks are concatenated.
    Unclosed thinking tags are treated as main text.
    """
    pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
    thinking_parts: list[str] = []
    main_parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > cursor:
            main_parts.append(text[cursor:start])
        thinking_parts.append(match.group(1).strip())
        cursor = end
    if cursor < len(text):
        main_parts.append(text[cursor:])
    return "\n\n".join(thinking_parts), "".join(main_parts).strip()


def _format_thinking(text: str) -> Text:
    """Render a thinking block as a dim italic indented block."""
    body = Text()
    for line in text.splitlines():
        body.append(f"  {_AGENT_GLYPH} ", style=_COLOR_MUTED)
        body.append(line, style=f"italic {_COLOR_DIM}")
        body.append("\n")
    return body


def _format_assistant_content(text: str, streaming: bool = False) -> RenderableType:
    """Return a renderable for assistant text with parsed thinking blocks."""
    thinking, main = _parse_thinking(text)
    parts: list[RenderableType] = []
    if thinking:
        parts.append(_format_thinking(thinking))
    if main or streaming:
        parts.append(Markdown(main) if main else Text("", style=_COLOR_MUTED))
    if not parts:
        return Text("", style=_COLOR_MUTED)
    if len(parts) == 1:
        return parts[0]
    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 0))
    for part in parts:
        table.add_row(part)
    return table


def _format_assistant_line(text: str, streaming: bool = False) -> Text:
    """Inline assistant prefix + content (Claude Code style)."""
    line = Text()
    line.append(f"{_AGENT_GLYPH} ", style=f"bold {_COLOR_YELLOW}")
    line.append(_format_assistant_content(text, streaming=streaming))
    return line


def _format_assistant_panel(text: str, streaming: bool = False) -> Panel:
    """Panel variant used for the final streamed message."""
    title = f"[bold {_COLOR_YELLOW}]Micro-Terse[/bold {_COLOR_YELLOW}]"
    if streaming:
        title += f" [{_COLOR_MUTED}]typing...[/]"
    return Panel(
        _format_assistant_content(text, streaming=streaming),
        title=title,
        title_align="left",
        border_style=_COLOR_GOLD,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _format_user_line(text: str) -> Text:
    """Compact user label + message in pale yellow."""
    line = Text()
    line.append(f"{_USER_GLYPH} ", style=f"bold {_COLOR_USER}")
    line.append("You ", style=f"bold {_COLOR_USER}")
    line.append(text, style="default")
    return line


def _format_header(model: str, url: str, online: bool, demo_mode: bool, block_on: bool = True) -> RenderableType:
    """Claude Code-style minimal header: logo + one-line status."""
    status_color = _COLOR_ONLINE if online else _COLOR_OFFLINE
    mode_color = _COLOR_GOLD if demo_mode else _COLOR_ONLINE
    mode_text = "demo" if demo_mode else "real"

    logo = _format_logo(block_on=block_on)

    status = Text()
    status.append(f"{model}", style=f"bold {_COLOR_YELLOW}")
    status.append("  ·  ", style=_COLOR_MUTED)
    status.append(url, style=_COLOR_DIM)
    status.append("  ·  ", style=_COLOR_MUTED)
    status.append("● ", style=f"bold {status_color}")
    status.append("online" if online else "offline", style=status_color)
    status.append("  ·  ", style=_COLOR_MUTED)
    status.append(mode_text, style=f"bold {mode_color}")

    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 0))
    table.add_row(logo)
    table.add_row(status)
    return table


def _format_status_panel(data: dict[str, Any]) -> Panel:
    mode = "demo" if data.get("demo_mode") else "real checkpoint"
    device = data.get("device", "unknown")
    model = data.get("model", "terse-micro")
    text = Text()
    text.append("Model:  ", style=f"bold {_COLOR_YELLOW}")
    text.append(f"{model}\n", style="default")
    text.append("Device: ", style=f"bold {_COLOR_YELLOW}")
    text.append(f"{device}\n", style="default")
    text.append("Mode:   ", style=f"bold {_COLOR_YELLOW}")
    text.append(mode, style="default")
    return Panel(text, title="[dim]status[/dim]", border_style=_COLOR_GOLD, box=box.ROUNDED)


def _format_models_panel(data: dict[str, Any]) -> Panel:
    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 0))
    for m in data.get("data", []):
        table.add_row(Text(f"{_AGENT_GLYPH} ", style=_COLOR_YELLOW), Text(m.get("id", ""), style="default"))
    return Panel(table, title="[dim]models[/dim]", border_style=_COLOR_GOLD, box=box.ROUNDED)


def _format_help_panel() -> Panel:
    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 0))
    table.add_row(Text("/help", style=f"bold {_COLOR_YELLOW}"), Text("show this help"))
    table.add_row(Text("/about", style=f"bold {_COLOR_YELLOW}"), Text("show the model spec sheet"))
    table.add_row(Text("/proof", style=f"bold {_COLOR_YELLOW}"), Text("prove identity alignment (live)"))
    table.add_row(Text("/predict", style=f"bold {_COLOR_YELLOW}"), Text("base-model next-token demo (+ free text)"))
    table.add_row(Text("/try N", style=f"bold {_COLOR_YELLOW}"), Text("run suggested prompt N"))
    table.add_row(Text("/status", style=f"bold {_COLOR_YELLOW}"), Text("show backend status"))
    table.add_row(Text("/models", style=f"bold {_COLOR_YELLOW}"), Text("list available models"))
    table.add_row(Text("/new", style=f"bold {_COLOR_YELLOW}"), Text("start a new conversation"))
    table.add_row(Text("/clear", style=f"bold {_COLOR_YELLOW}"), Text("clear the screen"))
    table.add_row(Text("/quit", style=f"bold {_COLOR_YELLOW}"), Text("exit"))
    return Panel(table, title="[dim]commands[/dim]", border_style=_COLOR_GOLD, box=box.ROUNDED)


def _format_welcome_panel() -> Panel:
    """Startup card: one-line intro + numbered starter prompts."""
    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
    intro = Text()
    intro.append("A 423M ", style=f"bold {_COLOR_YELLOW}")
    intro.append("ternary", style=f"bold {_COLOR_GOLD}")
    intro.append(" model, trained from scratch. Chat below — or run ", style="default")
    intro.append("/proof", style=f"bold {_COLOR_YELLOW}")
    intro.append(" to see it knows it's Terse:", style="default")
    table.add_row(intro)
    table.add_row(Text(""))
    for i, prompt in enumerate(_SUGGESTED_PROMPTS, start=1):
        row = Text()
        row.append(f"  {i} ", style=f"bold {_COLOR_YELLOW}")
        row.append(prompt, style="default")
        table.add_row(row)
    table.add_row(Text(""))
    hint = Text()
    hint.append("Chat, or:  ", style=_COLOR_MUTED)
    hint.append("/proof", style=f"bold {_COLOR_MUTED}")
    hint.append(" identity · ", style=_COLOR_MUTED)
    hint.append("/predict", style=f"bold {_COLOR_MUTED}")
    hint.append(" knowledge · ", style=_COLOR_MUTED)
    hint.append("/about", style=f"bold {_COLOR_MUTED}")
    hint.append(" specs", style=_COLOR_MUTED)
    table.add_row(hint)
    return Panel(
        table,
        title=f"[bold {_COLOR_YELLOW}]Welcome to Micro-Terse[/]",
        title_align="left",
        border_style=_COLOR_GOLD,
        box=box.ROUNDED,
        padding=(1, 1),
    )


def _format_about_panel() -> Panel:
    """Spec card shown by /about — the facts that make the model impressive."""
    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
    for label, value in _SPEC_FACTS:
        table.add_row(
            Text(f"{label}", style=f"bold {_COLOR_YELLOW}"),
            Text(value, style="default"),
        )
    return Panel(
        table,
        title=f"[bold {_COLOR_YELLOW}]Terse-Micro — spec sheet[/]",
        title_align="left",
        border_style=_COLOR_GOLD,
        box=box.ROUNDED,
        padding=(1, 1),
    )


def _format_metrics_line(elapsed: float, content: str) -> Text:
    """Dim one-line throughput readout shown after each response.

    Token count is approximate (~chars/4, the usual rough heuristic) — the CLI
    talks to the server over the OpenAI API and never sees the real token ids.
    """
    approx_tokens = max(1, len(content) // 4)
    tok_per_s = approx_tokens / elapsed if elapsed > 0 else 0.0
    line = Text()
    line.append("  ⚡ ", style=_COLOR_GOLD)
    line.append(f"{elapsed:.1f}s", style=_COLOR_DIM)
    line.append(" · ", style=_COLOR_MUTED)
    line.append(f"~{tok_per_s:.0f} tok/s", style=_COLOR_DIM)
    line.append(" · ", style=_COLOR_MUTED)
    line.append("423M ternary on CPU", style=_COLOR_MUTED)
    return line


def _format_margin_bar(margin: float, width: int = 10, span: float = 3.0) -> Text:
    """A small centered bar: green to the right when the model prefers Terse,
    red to the left when it prefers ChatGPT. `span` is the margin mapped to a
    full-width bar."""
    frac = max(-1.0, min(1.0, margin / span))
    filled = int(round(abs(frac) * width))
    bar = Text()
    if margin >= 0:
        bar.append(" " * width)
        bar.append("│", style=_COLOR_MUTED)
        bar.append("█" * filled, style=f"bold {_COLOR_ONLINE}")
        bar.append(" " * (width - filled))
    else:
        bar.append(" " * (width - filled))
        bar.append("█" * filled, style=f"bold {_COLOR_OFFLINE}")
        bar.append("│", style=_COLOR_MUTED)
        bar.append(" " * width)
    return bar


def _format_progression(stages: list[dict[str, Any]]) -> Table:
    """Compact base -> SFT -> ORPO average-margin progression."""
    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
    for st in stages:
        avg = st.get("avg_margin", 0.0)
        won = st.get("preferred", 0)
        total = st.get("total", 0)
        color = _COLOR_ONLINE if avg >= 0 else _COLOR_OFFLINE
        row = Text()
        row.append(f"{st.get('stage', '?'):>5} ", style=f"bold {_COLOR_YELLOW}")
        row.append(_format_margin_bar(avg))
        row.append(f"  {avg:+.2f}", style=color)
        row.append(f"   {won}/{total} prefer Terse", style=_COLOR_DIM)
        table.add_row(row)
    return table


def _format_proof_panel(data: dict[str, Any]) -> Panel:
    """Render the live identity-preference proof (+ training progression if present)."""
    if not data.get("available"):
        body = Text()
        body.append("The identity proof needs the real model loaded.\n", style="default")
        body.append("The backend is in demo mode — start it with the trained GGUF.", style=_COLOR_MUTED)
        return Panel(
            body,
            title=f"[bold {_COLOR_YELLOW}]Identity proof[/]",
            title_align="left",
            border_style=_COLOR_GOLD,
            box=box.ROUNDED,
            padding=(1, 1),
        )

    probes = data.get("probes", [])
    total = data.get("total", len(probes))
    preferred = data.get("preferred", 0)

    headline = Text()
    won_all = preferred == total and total > 0
    headline.append("Live on the loaded model: ", style=_COLOR_DIM)
    headline.append(
        f"prefers being Terse on {preferred}/{total} probes",
        style=f"bold {_COLOR_ONLINE if won_all else _COLOR_YELLOW}",
    )

    inner = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
    for p in probes:
        margin = p.get("margin", 0.0)
        prefers = p.get("prefers_charter", margin >= 0)
        color = _COLOR_ONLINE if prefers else _COLOR_OFFLINE
        inner.add_row(
            Text(p.get("question", ""), style="default"),
            _format_margin_bar(margin),
            Text(f"{margin:+.2f}", style=color),
            Text("Terse" if prefers else "ChatGPT", style=color),
        )

    note = Text()
    note.append(
        "margin = log-prob it gives the Terse answer minus the ChatGPT answer; "
        "positive = it prefers being Terse.",
        style=_COLOR_MUTED,
    )

    outer = Table(box=None, show_header=False, show_edge=False, padding=(0, 0))
    outer.add_row(headline)
    outer.add_row(Text(""))
    outer.add_row(inner)
    progression = data.get("progression")
    if progression:
        outer.add_row(Text(""))
        outer.add_row(Text("How training moved it (base → SFT → ORPO):", style=f"bold {_COLOR_GOLD}"))
        outer.add_row(_format_progression(progression))
    outer.add_row(Text(""))
    outer.add_row(note)

    return Panel(
        outer,
        title=f"[bold {_COLOR_YELLOW}]Identity proof — does it prefer being Terse?[/]",
        title_align="left",
        border_style=_COLOR_GOLD,
        box=box.ROUNDED,
        padding=(1, 1),
    )


def _prob_bar(prob: float, width: int = 16) -> Text:
    """Green proportional bar for a probability in [0, 1]."""
    filled = int(round(max(0.0, min(1.0, prob)) * width))
    bar = Text()
    bar.append("█" * filled, style=f"bold {_COLOR_ONLINE}")
    bar.append("·" * (width - filled), style=_COLOR_MUTED)
    return bar


def _format_prediction_panel(data: dict[str, Any]) -> Panel:
    """Render a single next-token prediction (POST /v1/predict)."""
    if not data.get("available"):
        body = Text()
        body.append("Next-token prediction needs the base model loaded.\n", style="default")
        body.append("The backend is in demo mode.", style=_COLOR_MUTED)
        return Panel(body, title=f"[bold {_COLOR_YELLOW}]Predict[/]", title_align="left",
                     border_style=_COLOR_GOLD, box=box.ROUNDED, padding=(1, 1))

    prompt = Text()
    prompt.append(data.get("text", ""), style="default")
    prompt.append(" ", style="default")
    prompt.append("___", style=f"bold {_COLOR_YELLOW}")

    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
    for i, pred in enumerate(data.get("predictions", [])):
        token = (pred.get("token") or "").strip() or "·"
        prob = pred.get("prob", 0.0)
        color = f"bold {_COLOR_ONLINE}" if i == 0 else _COLOR_DIM
        table.add_row(Text(token, style=color), _prob_bar(prob), Text(f"{prob:.0%}", style=color))

    outer = Table(box=None, show_header=False, show_edge=False, padding=(0, 0))
    outer.add_row(prompt)
    outer.add_row(Text(""))
    outer.add_row(table)
    outer.add_row(Text(""))
    outer.add_row(Text("The base model's next-token guess — learned weights, no lookup.", style=_COLOR_MUTED))
    return Panel(outer, title=f"[bold {_COLOR_YELLOW}]Next-token prediction (base model)[/]",
                 title_align="left", border_style=_COLOR_GOLD, box=box.ROUNDED, padding=(1, 1))


def _format_showcase_panel(data: dict[str, Any]) -> Panel:
    """Render the curated next-token showcase (GET /v1/predict_showcase)."""
    if not data.get("available"):
        body = Text()
        body.append("The base-model showcase needs the base model loaded.\n", style="default")
        body.append("The backend is in demo mode.", style=_COLOR_MUTED)
        return Panel(body, title=f"[bold {_COLOR_YELLOW}]Base model showcase[/]", title_align="left",
                     border_style=_COLOR_GOLD, box=box.ROUNDED, padding=(1, 1))

    table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
    for item in data.get("items", []):
        preds = item.get("predictions", [])
        top = preds[0] if preds else {"token": "", "prob": 0.0}
        token = (top.get("token") or "").strip() or "·"
        prob = top.get("prob", 0.0)
        line = Text()
        line.append(item.get("text", ""), style=_COLOR_DIM)
        line.append("  →  ", style=_COLOR_MUTED)
        line.append(token, style=f"bold {_COLOR_ONLINE}")
        table.add_row(line, Text(f"{prob:.0%}", style=_COLOR_YELLOW))

    outer = Table(box=None, show_header=False, show_edge=False, padding=(0, 0))
    outer.add_row(Text("The 423M base model completes these correctly — facts in 182 MB:",
                       style="default"))
    outer.add_row(Text(""))
    outer.add_row(table)
    outer.add_row(Text(""))
    outer.add_row(Text("Try your own: /predict The capital of France is", style=_COLOR_MUTED))
    return Panel(outer, title=f"[bold {_COLOR_YELLOW}]What the base model knows — next-token prediction[/]",
                 title_align="left", border_style=_COLOR_GOLD, box=box.ROUNDED, padding=(1, 1))


def _format_footer() -> Text:
    footer = Text()
    footer.append("Micro-Terse ", style=_COLOR_MUTED)
    footer.append("· ", style=_COLOR_MUTED)
    footer.append("/help ", style=f"bold {_COLOR_MUTED}")
    footer.append("commands · ", style=_COLOR_MUTED)
    footer.append("Ctrl+C ", style=f"bold {_COLOR_MUTED}")
    footer.append("quit", style=_COLOR_MUTED)
    return footer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class MicroTerseCLI:
    def __init__(
        self,
        url: str,
        api_key: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool,
        console: Console | None = None,
    ) -> None:
        if not _HAS_RICH:
            raise RuntimeError(
                "The styled CLI requires 'rich'. Install it with: pip install -e '.[cli]'"
            )
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.console = console or Console(highlight=False)
        self.messages: list[dict[str, str]] = []
        self._online = True
        self._block_on = True
        self._animation_stop = threading.Event()
        self._animation_thread: threading.Thread | None = None


    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> dict | None:
        try:
            r = httpx.get(f"{self.url}{path}", headers=self._headers(), timeout=10.0)
            r.raise_for_status()
            self._online = True
            return r.json()
        except Exception as exc:
            self._online = False
            self.console.print(self._format_error(str(exc)))
            return None

    def _print_header(self) -> None:
        data = self._get("/v1/status") or {}
        demo_mode = data.get("demo_mode", True)
        self.console.print(_format_header(self.model, self.url, self._online, demo_mode, self._block_on))
        self.console.print(_format_footer())
        self.console.print()

    def _animation_loop(self) -> None:
        while not self._animation_stop.is_set():
            time.sleep(1.0)
            if self._animation_stop.is_set():
                break
            self._block_on = not self._block_on
            try:
                self._redraw_logo_line()
            except Exception:
                break

    def _redraw_logo_line(self) -> None:
        """Redraw only the animated block cursor in the logo line.

        We use raw ANSI escapes rather than ``console.print`` so Rich does not
        buffer or relocate the character, which caused the duplication bug.
        The cursor is positioned at the exact cell of the block (row 3, col 6),
        then a gold-colored marker or space is written. The cursor is then
        returned to its previous position so subsequent input is not drawn over
        the logo.
        """
        marker = "▌" if self._block_on else " "
        colored_marker = f"{_ansi_fg(_COLOR_GOLD)}{marker}\033[0m"
        # Save cursor position, jump to the logo cursor, update it, restore position.
        sys.stdout.write(
            f"\033[s\033[{_LOGO_CURSOR_ROW};{_LOGO_CURSOR_COL}H{colored_marker}\033[u"
        )
        sys.stdout.flush()

    def _start_animation(self) -> None:
        self._animation_stop.clear()
        self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self._animation_thread.start()

    def _stop_animation(self) -> None:
        self._animation_stop.set()
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=1.0)

    def _post(self, path: str, body: dict) -> dict | None:
        try:
            r = httpx.post(f"{self.url}{path}", headers=self._headers(), json=body, timeout=60.0)
            r.raise_for_status()
            self._online = True
            return r.json()
        except Exception as exc:
            self._online = False
            self.console.print(self._format_error(str(exc)))
            return None

    def _print_status(self) -> None:
        data = self._get("/v1/status")
        if data:
            self.console.print(_format_status_panel(data))

    def _print_models(self) -> None:
        data = self._get("/v1/models")
        if data:
            self.console.print(_format_models_panel(data))

    def _print_help(self) -> None:
        self.console.print(_format_help_panel())

    def _print_about(self) -> None:
        self.console.print(_format_about_panel())

    def _print_proof(self) -> None:
        with Status(
            "[dim]Scoring identity probes on the model...[/]",
            spinner=_SPINNER,
            console=self.console,
            speed=1.5,
        ):
            data = self._get("/v1/identity_proof")
        if data is not None:
            self.console.print(_format_proof_panel(data))

    def _print_welcome(self) -> None:
        self.console.print(_format_welcome_panel())
        self.console.print()

    def _print_predict(self, arg: str) -> None:
        """/predict -> curated base-model showcase; /predict <text> -> live."""
        arg = arg.strip()
        with Status(
            "[dim]Predicting next token on the base model...[/]",
            spinner=_SPINNER,
            console=self.console,
            speed=1.5,
        ):
            if arg:
                data = self._post("/v1/predict", {"text": arg, "k": 5})
                panel = _format_prediction_panel(data) if data is not None else None
            else:
                data = self._get("/v1/predict_showcase")
                panel = _format_showcase_panel(data) if data is not None else None
        if panel is not None:
            self.console.print(panel)

    def _try_suggestion(self, arg: str) -> None:
        """Run one of the numbered starter prompts (/try N)."""
        try:
            n = int(arg)
        except ValueError:
            self.console.print(self._format_error("Usage: /try N  (e.g. /try 1)"))
            return
        if not 1 <= n <= len(_SUGGESTED_PROMPTS):
            self.console.print(
                self._format_error(f"/try expects a number 1-{len(_SUGGESTED_PROMPTS)}")
            )
            return
        self._send(_SUGGESTED_PROMPTS[n - 1])

    def _clear_screen(self) -> None:
        self.console.clear()
        self._print_header()

    def _format_error(self, message: str) -> Text:
        return Text(f"Error: {message}", style=f"bold {_COLOR_ERROR}")

    def _send(self, text: str) -> None:
        self.console.print(_format_user_line(text))
        self.messages.append({"role": "user", "content": text})

        started = time.perf_counter()
        try:
            if self.stream:
                content = self._stream_response()
            else:
                content = self._fetch_response()
        except Exception as exc:
            self.console.print(self._format_error(str(exc)))
            if self.messages:
                self.messages.pop()
            return
        elapsed = time.perf_counter() - started

        self.messages.append({"role": "assistant", "content": content})
        if content.strip():
            self.console.print(_format_metrics_line(elapsed, content))
        self.console.print()

    def _request_body(self) -> dict:
        return {
            "model": self.model,
            "messages": self.messages,
            "stream": self.stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def _fetch_response(self) -> str:
        with Status(
            "[dim]Micro-Terse is thinking...[/]",
            spinner=_SPINNER,
            console=self.console,
            speed=1.5,
        ):
            r = httpx.post(
                f"{self.url}/v1/chat/completions",
                headers=self._headers(),
                json=self._request_body(),
                timeout=120.0,
            )
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]
        self.console.print(_format_assistant_panel(content))
        return content

    def _stream_response(self) -> str:
        full = ""
        panel = _format_assistant_panel(full, streaming=True)
        with Live(panel, console=self.console, refresh_per_second=12, auto_refresh=False) as live:
            with httpx.stream(
                "POST",
                f"{self.url}/v1/chat/completions",
                headers=self._headers(),
                json=self._request_body(),
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        self.console.print(
                            Text("[malformed server chunk, skipping]", style=_COLOR_DIM)
                        )
                        continue
                    choices = parsed.get("choices") or [{}]
                    first = choices[0] or {}
                    delta = first.get("delta", {}).get("content", "")
                    if delta:
                        full += delta
                        live.update(_format_assistant_panel(full, streaming=True))

        # Persist the final rendered panel once streaming finishes.
        self.console.print(_format_assistant_panel(full, streaming=False))
        return full

    def run(self) -> int:
        self.console.show_cursor(False)
        self._print_header()
        self._print_welcome()
        self._start_animation()

        try:
            while True:
                try:
                    prompt = self.console.input(f"[{_COLOR_YELLOW}]❯[/] ")
                except (EOFError, KeyboardInterrupt):
                    self.console.print()
                    break

                text = prompt.strip()
                if not text:
                    continue
                if text in ("/quit", "/exit", "/q"):
                    break
                if text == "/help":
                    self._print_help()
                    continue
                if text == "/about":
                    self._print_about()
                    continue
                if text == "/proof":
                    self._print_proof()
                    continue
                if text == "/predict" or text.startswith("/predict "):
                    self._print_predict(text[len("/predict"):])
                    continue
                if text == "/try" or text.startswith("/try "):
                    self._try_suggestion(text[len("/try"):].strip())
                    continue
                if text == "/new":
                    self.messages.clear()
                    self.console.print(Text("New conversation started.", style=_COLOR_MUTED))
                    self.console.print()
                    continue
                if text == "/clear":
                    self._clear_screen()
                    continue
                if text == "/status":
                    self._print_status()
                    continue
                if text == "/models":
                    self._print_models()
                    continue
                if text.startswith("/"):
                    self.console.print(self._format_error(f"Unknown command: {text}"))
                    continue

                self._send(text)
        finally:
            self._stop_animation()
            self.console.show_cursor(True)

        self.console.print(Text("Goodbye.", style=_COLOR_MUTED))
        return 0


def _configure_stdio_encoding() -> None:
    """Ensure stdout/stderr use UTF-8 on Windows before Rich writes anything."""
    if os.name != "nt":
        return
    try:
        import sys

        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Micro-Terse terminal demo")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Backend base URL")
    parser.add_argument("--api-key", default=os.getenv("TERSE_API_KEY"), help="Bearer token")
    parser.add_argument("--model", default="terse-micro", help="Model id")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    args = parser.parse_args()

    if not _HAS_RICH:
        print(
            "Error: rich is required for the styled CLI. Run: pip install -e '.[cli]'",
            file=sys.stderr,
        )
        return 1

    _configure_windows_console()
    _configure_stdio_encoding()

    cli = MicroTerseCLI(
        url=args.url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stream=not args.no_stream,
    )
    return cli.run()


if __name__ == "__main__":
    raise SystemExit(main())
