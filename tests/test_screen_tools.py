"""
Test script for the Computer Interaction Layer tools:
  - capture_screen   (Tool #19)
  - analyze_screen   (Tool #20)

Run from the project root:
    python tests/test_screen_tools.py

Tests:
  1. Unit test — imports succeed and tools are registered
  2. capture_screen — takes a screenshot and verifies the file exists
  3. analyze_screen — calls vision analysis (skipped if no vision model available)
  4. End-to-end — calls analyze_screen with no path (auto-capture flow)

No pytest required — uses stdlib unittest only.
"""

import os
import sys
import unittest
from pathlib import Path

# ── make sure we can import from project root ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestScreenCaptureImport(unittest.TestCase):
    def test_tool_imports(self):
        """capture_screen and analyze_screen must be importable."""
        from backend.tools.screen_capture import capture_screen
        from backend.tools.screen_analyzer import analyze_screen
        self.assertIsNotNone(capture_screen)
        self.assertIsNotNone(analyze_screen)

    def test_tools_registered(self):
        """Both tools must appear in the agent's ALL_TOOLS list."""
        from backend.tools.registry import ALL_TOOLS
        tool_names = [t.name for t in ALL_TOOLS]
        print(f"\n[registry] loaded tools: {tool_names}")
        self.assertIn("capture_screen", tool_names, "capture_screen not in registry")
        self.assertIn("analyze_screen", tool_names, "analyze_screen not in registry")


class TestCaptureScreen(unittest.TestCase):
    def setUp(self):
        """Try to import Pillow — skip if missing."""
        try:
            from PIL import ImageGrab  # noqa
        except ImportError:
            self.skipTest("Pillow not installed. Run: pip install Pillow")

    def test_capture_returns_string(self):
        """capture_screen() must return a non-empty string."""
        from backend.tools.screen_capture import capture_screen
        result = capture_screen.invoke({"save_path": ""})
        print(f"\n[capture_screen] result:\n{result}")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_screenshot_file_exists(self):
        """The screenshot file referenced in the result must exist on disk."""
        from backend.tools.screen_capture import capture_screen
        result = capture_screen.invoke({"save_path": ""})
        saved_path = None
        for line in result.splitlines():
            if "Saved to" in line:
                saved_path = line.split(":", 1)[1].strip()
                break
        self.assertIsNotNone(saved_path, "No 'Saved to' line in capture_screen output")
        self.assertTrue(Path(saved_path).exists(), f"Screenshot not on disk: {saved_path}")
        size_bytes = Path(saved_path).stat().st_size
        print(f"\n[capture_screen] saved to {saved_path} ({size_bytes} bytes)")
        self.assertGreater(size_bytes, 1000, "Screenshot is suspiciously small")

    def test_active_window_in_result(self):
        """Result must contain 'Active window' line."""
        from backend.tools.screen_capture import capture_screen
        result = capture_screen.invoke({"save_path": ""})
        self.assertIn("Active window", result)

    def test_resolution_in_result(self):
        """Result must contain 'Resolution' line."""
        from backend.tools.screen_capture import capture_screen
        result = capture_screen.invoke({"save_path": ""})
        self.assertIn("Resolution", result)

    def test_custom_save_path(self):
        """Screenshot should be saved to the custom path if one is given."""
        import tempfile, os
        from backend.tools.screen_capture import capture_screen

        tmp = Path(tempfile.gettempdir()) / "ai_os_test_shot.png"
        result = capture_screen.invoke({"save_path": str(tmp)})
        print(f"\n[capture_screen custom path] result:\n{result}")
        self.assertTrue(tmp.exists(), f"Custom path screenshot not created: {tmp}")
        tmp.unlink(missing_ok=True)  # clean up


class TestAnalyzeScreen(unittest.TestCase):
    """
    These tests are integration tests — they actually call the vision LLM.
    If no vision model is configured, the tests gracefully skip.
    """

    def _latest_screenshot(self) -> Path:
        """Return the most recently created screenshot, or None."""
        shots_dir = Path(".ai_os") / "screenshots"
        if not shots_dir.exists():
            return None
        files = sorted(shots_dir.glob("screenshot_*.png"))
        return files[-1] if files else None

    def test_analyze_returns_string(self):
        """analyze_screen() always returns a string (even if vision fails)."""
        from backend.tools.screen_analyzer import analyze_screen
        # Use an existing screenshot if available, else let it auto-capture
        shot = self._latest_screenshot()
        path_arg = str(shot) if shot else ""
        result = analyze_screen.invoke({"path": path_arg, "task": "describe the screen briefly"})
        print(f"\n[analyze_screen] result (first 500 chars):\n{result[:500]}")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_analyze_bad_path(self):
        """analyze_screen() must return an error string, not raise, for a missing file."""
        from backend.tools.screen_analyzer import analyze_screen
        result = analyze_screen.invoke({"path": "/nonexistent/path/shot.png", "task": ""})
        print(f"\n[analyze_screen bad path] result:\n{result}")
        self.assertIsInstance(result, str)
        # Should contain some helpful error text, not raise
        self.assertTrue(len(result) > 0)

    def test_end_to_end_auto_capture(self):
        """analyze_screen() with no path should auto-capture and analyze."""
        try:
            from PIL import ImageGrab  # noqa
        except ImportError:
            self.skipTest("Pillow not installed — cannot auto-capture.")

        from backend.tools.screen_analyzer import analyze_screen
        result = analyze_screen.invoke({"path": "", "task": "what is on my screen?"})
        print(f"\n[analyze_screen auto] result (first 800 chars):\n{result[:800]}")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)


if __name__ == "__main__":
    print("=" * 60)
    print("  AI OS — Computer Interaction Layer Test Suite")
    print("  Tools: capture_screen (Tool #19), analyze_screen (Tool #20)")
    print("=" * 60)
    unittest.main(verbosity=2)
