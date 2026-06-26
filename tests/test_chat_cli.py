"""Tests for the terminal chat CLI."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from terse.cli import chat as chat_module
from terse.cli.chat import (
    MicroTerseCLI,
    _configure_windows_console,
    _format_assistant_content,
    _format_assistant_panel,
    _format_footer,
    _format_header,
    _format_help_panel,
    _format_logo,
    _format_models_panel,
    _format_status_panel,
    _format_thinking,
    _format_user_line,
    _parse_thinking,
    main,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False


pytestmark = pytest.mark.skipif(not _HAS_RICH, reason="rich not installed")


def _make_console() -> Console:
    return Console(record=True, force_terminal=True, width=120, highlight=False)


def _capture_renderable(renderable) -> str:
    console = _make_console()
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _status_dict() -> dict:
    return {"demo_mode": True, "device": "cpu", "model": "terse-micro"}


def _make_cli(**kwargs) -> MicroTerseCLI:
    defaults = {
        "url": "http://127.0.0.1:8080",
        "api_key": None,
        "model": "terse-micro",
        "temperature": 0.7,
        "max_tokens": 64,
        "top_p": 0.9,
        "stream": True,
        "console": _make_console(),
    }
    defaults.update(kwargs)
    return MicroTerseCLI(**defaults)


# ---------------------------------------------------------------------------
# Rendering helper tests
# ---------------------------------------------------------------------------
def test_parse_thinking_extracts_blocks():
    thinking, main = _parse_thinking("hello <thinking>reason</thinking> world")
    assert thinking == "reason"
    assert main == "hello  world"


def test_parse_thinking_multiple_blocks():
    thinking, main = _parse_thinking("a <thinking>x</thinking> b <thinking>y</thinking> c")
    assert thinking == "x\n\ny"
    assert main == "a  b  c"


def test_parse_thinking_no_blocks():
    thinking, main = _parse_thinking("plain text")
    assert thinking == ""
    assert main == "plain text"


def test_parse_thinking_unclosed_tag():
    thinking, main = _parse_thinking("start <thinking> no end")
    assert thinking == ""
    assert main == "start <thinking> no end"


def test_format_thinking_renders_reasoning():
    panel = _format_thinking("reason")
    rendered = _capture_renderable(panel)
    assert "reason" in rendered
    assert "✻" in rendered




def test_format_assistant_content_renders_main_text():
    renderable = _format_assistant_content("answer <thinking>reason</thinking> more")
    rendered = _capture_renderable(renderable)
    assert "answer" in rendered
    assert "more" in rendered
    assert "reason" in rendered


def test_format_assistant_panel_renders_label():
    panel = _format_assistant_panel("hello")
    rendered = _capture_renderable(panel)
    assert "Micro-Terse" in rendered
    assert "hello" in rendered


def test_format_user_line_contains_message():
    text = _format_user_line("hello")
    assert "hello" in str(text)


def test_format_header_contains_model_and_url():
    panel = _format_header("terse-micro", "http://test", True, True)
    rendered = _capture_renderable(panel)
    assert "╭───────╮" in rendered
    assert "t ▌" in rendered
    assert "terse-micro" in rendered
    assert "http://test" in rendered
    assert "demo" in rendered


def test_format_status_panel_contains_mode():
    panel = _format_status_panel({"demo_mode": True, "device": "cpu", "model": "terse-micro"})
    rendered = _capture_renderable(panel)
    assert "demo" in rendered
    assert "cpu" in rendered


def test_format_models_panel_contains_ids():
    panel = _format_models_panel({"data": [{"id": "terse-micro"}]})
    rendered = _capture_renderable(panel)
    assert "terse-micro" in rendered


def test_format_help_panel_contains_commands():
    panel = _format_help_panel()
    rendered = _capture_renderable(panel)
    assert "/status" in rendered
    assert "/help" in rendered
    assert "/new" in rendered


def test_format_footer_contains_hints():
    text = _format_footer()
    rendered = str(text)
    assert "/help" in rendered
    assert "Ctrl+C" in rendered


def test_format_error_method():
    cli = _make_cli()
    text = cli._format_error("boom")
    assert "boom" in str(text)


# ---------------------------------------------------------------------------
# CLI API tests
# ---------------------------------------------------------------------------
def test_cli_headers_include_api_key():
    cli = _make_cli(api_key="secret")
    assert cli._headers()["Authorization"] == "Bearer secret"


def test_cli_headers_without_api_key():
    cli = _make_cli()
    assert "Authorization" not in cli._headers()


def test_request_body_shape():
    cli = _make_cli()
    cli.messages = [{"role": "user", "content": "hi"}]
    body = cli._request_body()
    assert body["model"] == "terse-micro"
    assert body["stream"] is True
    assert body["messages"][-1]["content"] == "hi"


def test_get_returns_json():
    cli = _make_cli()
    fake = MagicMock()
    fake.json.return_value = {"demo_mode": True, "device": "cpu", "model": "terse-micro"}
    fake.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=fake) as mock_get:
        result = cli._get("/v1/status")
        assert result["device"] == "cpu"
        assert cli._online is True
        mock_get.assert_called_with(
            "http://127.0.0.1:8080/v1/status", headers=cli._headers(), timeout=10.0
        )


def test_get_returns_none_on_error():
    cli = _make_cli()
    with patch("httpx.get", side_effect=RuntimeError("boom")):
        assert cli._get("/v1/status") is None
        assert cli._online is False


def test_print_status_with_mocked_get(capsys):
    cli = _make_cli()
    with patch.object(cli, "_get", return_value=_status_dict()):
        cli._print_status()
    assert cli._online is True


def test_print_models_with_mocked_get(capsys):
    cli = _make_cli()
    with patch.object(cli, "_get", return_value={"data": [{"id": "terse-micro"}]}):
        cli._print_models()


def test_fetch_response_parses_non_streaming():
    cli = _make_cli(stream=False)
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=fake_response) as mock_post:
        content = cli._fetch_response()
        assert content == "Hello"
        mock_post.assert_called_once()
        assert len(cli.messages) == 0


def _fake_stream(lines: list[str]):
    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self):
            for line in lines:
                yield line

        def raise_for_status(self):
            pass

    return FakeStream()


def test_stream_response_parses_sse_chunks():
    cli = _make_cli()
    stream = _fake_stream([
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        "data: [DONE]",
    ])
    with patch("httpx.stream", return_value=stream) as mock_stream:
        content = cli._stream_response()
        assert content == "Hi"
        mock_stream.assert_called_once()


def test_stream_response_handles_empty_choices():
    cli = _make_cli()
    stream = _fake_stream([
        'data: {"choices":[]}',
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        "data: [DONE]",
    ])
    with patch("httpx.stream", return_value=stream):
        content = cli._stream_response()
        assert content == "Hi"


def test_stream_response_warns_on_malformed_json():
    cli = _make_cli()
    stream = _fake_stream([
        "data: not-json",
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        "data: [DONE]",
    ])
    with patch("httpx.stream", return_value=stream):
        content = cli._stream_response()
        assert content == "Hi"


def test_send_stores_messages_and_calls_stream():
    cli = _make_cli()
    with patch.object(cli, "_stream_response", return_value="Hello"):
        cli._send("hi")
    assert len(cli.messages) == 2
    assert cli.messages[0]["role"] == "user"
    assert cli.messages[1]["role"] == "assistant"


def test_send_recovers_on_error():
    cli = _make_cli()
    with patch.object(cli, "_stream_response", side_effect=RuntimeError("boom")):
        cli._send("hi")
    assert len(cli.messages) == 0


def test_send_error_guard_when_messages_empty():
    cli = _make_cli()
    cli.messages = []
    with patch.object(cli, "_stream_response", side_effect=RuntimeError("boom")):
        cli._send("hi")  # should not raise


# ---------------------------------------------------------------------------
# Command loop tests
# ---------------------------------------------------------------------------
def _run_with_commands(*commands: str) -> tuple[int, MicroTerseCLI]:
    cli = _make_cli()
    with patch.object(cli, "_get", return_value=_status_dict()):
        with patch.object(cli.console, "input", side_effect=list(commands)):
            code = cli.run()
    return code, cli


def test_run_quit_command():
    cli = _make_cli()
    with patch.object(cli, "_get", return_value=_status_dict()), patch.object(
        cli.console, "input", side_effect=["/quit"]
    ):
        assert cli.run() == 0


def test_run_help_then_quit():
    code, _ = _run_with_commands("/help", "/quit")
    assert code == 0


def test_run_about_then_quit():
    code, _ = _run_with_commands("/about", "/quit")
    assert code == 0


def test_run_proof_then_quit():
    code, _ = _run_with_commands("/proof", "/quit")
    assert code == 0


def test_run_predict_showcase_then_quit():
    code, _ = _run_with_commands("/predict", "/quit")
    assert code == 0


def test_run_predict_text_then_quit():
    cli = _make_cli()
    payload = {"available": True, "text": "The sky is", "predictions": [{"token": " blue", "prob": 0.4}]}
    with patch.object(cli, "_get", return_value=_status_dict()), patch.object(
        cli, "_post", return_value=payload
    ) as mock_post, patch.object(cli.console, "input", side_effect=["/predict The sky is", "/quit"]):
        assert cli.run() == 0
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == "/v1/predict"


def test_format_prediction_panel():
    from terse.cli.chat import _format_prediction_panel

    panel = _format_prediction_panel(
        {
            "available": True,
            "text": "The sky is",
            "predictions": [{"token": " blue", "prob": 0.4}, {"token": " the", "prob": 0.2}],
        }
    )
    assert panel is not None


def test_format_showcase_panel():
    from terse.cli.chat import _format_showcase_panel

    panel = _format_showcase_panel(
        {"available": True, "items": [{"text": "X", "predictions": [{"token": " Y", "prob": 0.9}]}]}
    )
    assert panel is not None


def test_format_prediction_panel_unavailable():
    from terse.cli.chat import _format_prediction_panel

    assert _format_prediction_panel({"available": False, "predictions": []}) is not None


def test_format_proof_panel_unavailable():
    from terse.cli.chat import _format_proof_panel

    panel = _format_proof_panel({"available": False, "probes": [], "preferred": 0, "total": 0})
    assert panel is not None


def test_format_proof_panel_with_progression():
    from terse.cli.chat import _format_proof_panel

    data = {
        "available": True,
        "total": 2,
        "preferred": 1,
        "probes": [
            {"question": "Who are you?", "margin": 1.5, "prefers_charter": True},
            {"question": "Are you ChatGPT?", "margin": -0.7, "prefers_charter": False},
        ],
        "progression": [
            {"stage": "base", "avg_margin": -1.8, "preferred": 0, "total": 4},
            {"stage": "orpo", "avg_margin": 0.9, "preferred": 3, "total": 4},
        ],
    }
    panel = _format_proof_panel(data)
    assert panel is not None


def test_margin_bar_direction():
    from terse.cli.chat import _format_margin_bar

    pos = _format_margin_bar(2.0)
    neg = _format_margin_bar(-2.0)
    # Both render to the same fixed width; sign only changes which side fills.
    assert pos.cell_len == neg.cell_len


def test_try_suggestion_sends_curated_prompt():
    from terse.cli.chat import _SUGGESTED_PROMPTS

    cli = _make_cli()
    with patch.object(cli, "_get", return_value=_status_dict()), patch.object(
        cli, "_send"
    ) as mock_send, patch.object(cli.console, "input", side_effect=["/try 2", "/quit"]):
        assert cli.run() == 0
    mock_send.assert_called_once_with(_SUGGESTED_PROMPTS[1])


def test_try_suggestion_rejects_out_of_range():
    cli = _make_cli()
    with patch.object(cli, "_get", return_value=_status_dict()), patch.object(
        cli, "_send"
    ) as mock_send, patch.object(cli.console, "input", side_effect=["/try 99", "/quit"]):
        assert cli.run() == 0
    mock_send.assert_not_called()


def test_run_new_then_quit():
    cli = _make_cli()
    cli.messages = [{"role": "user", "content": "old"}]
    with patch.object(cli, "_get", return_value=_status_dict()), patch.object(
        cli.console, "input", side_effect=["/new", "/quit"]
    ):
        assert cli.run() == 0
        assert cli.messages == []


def test_run_status_then_quit():
    cli = _make_cli()
    with patch.object(cli, "_get", return_value=_status_dict()) as mock_get, patch.object(
        cli.console, "input", side_effect=["/status", "/quit"]
    ):
        assert cli.run() == 0
        # _get is called by _print_header and by the /status command. Startup now
        # shows the welcome panel instead of dumping status, so it's 2, not 3.
        assert mock_get.call_count == 2


def test_run_models_then_quit():
    code, _ = _run_with_commands("/models", "/quit")
    assert code == 0


def test_run_clear_then_quit():
    code, _ = _run_with_commands("/clear", "/quit")
    assert code == 0


def test_run_unknown_command():
    code, _ = _run_with_commands("/foo", "/quit")
    assert code == 0


def test_run_sends_message():
    cli = _make_cli()
    with patch.object(cli, "_get", return_value=_status_dict()), patch.object(
        cli.console, "input", side_effect=["hello", "/quit"]
    ), patch.object(cli, "_send") as mock_send:
        assert cli.run() == 0
        mock_send.assert_called_once_with("hello")


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------
def test_main_argparse_with_no_stream():
    with patch("sys.argv", ["chat.py", "--url", "http://test", "--no-stream"]), patch.object(
        MicroTerseCLI, "run", return_value=0
    ) as mock_run:
        assert main() == 0
        mock_run.assert_called_once()


def test_main_argparse():
    with patch("sys.argv", ["chat.py", "--url", "http://test"]), patch.object(
        MicroTerseCLI, "run", return_value=0
    ):
        assert main() == 0


def test_configure_windows_console_runs_on_nt():
    with patch("os.name", "nt"):
        with patch("ctypes.windll.kernel32.SetConsoleCP") as mock_cp, patch(
            "ctypes.windll.kernel32.SetConsoleOutputCP"
        ) as mock_out:
            _configure_windows_console()
            mock_cp.assert_called_once_with(65001)
            mock_out.assert_called_once_with(65001)


def test_configure_windows_console_skips_non_nt():
    with patch("os.name", "posix"):
        # Should complete without touching ctypes.
        _configure_windows_console()


def test_main_requires_rich(monkeypatch, capsys):
    monkeypatch.setattr(chat_module, "_HAS_RICH", False)
    with patch("sys.argv", ["chat.py", "--url", "http://test"]):
        assert main() == 1
    captured = capsys.readouterr()
    assert "rich is required" in captured.err


def test_cli_requires_rich():
    with patch.object(chat_module, "_HAS_RICH", False):
        with pytest.raises(RuntimeError, match="rich"):
            MicroTerseCLI(
                url="http://test",
                api_key=None,
                model="terse-micro",
                temperature=0.7,
                max_tokens=64,
                top_p=0.9,
                stream=True,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
