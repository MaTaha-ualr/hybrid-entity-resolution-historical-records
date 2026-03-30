"""Driver to execute HM_Taha notebook cells as standalone modules."""

from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import logging
import runpy
import sys
import types
from datetime import datetime
from pathlib import Path

MODULES_DIR = Path(__file__).parent / "hm_taha_modules"
ORIGINAL_OUTPUTS_DIR = MODULES_DIR / "original_outputs"
RUNTIME_OUTPUTS_ROOT = MODULES_DIR / "runtime_outputs"
LOG_DIR = Path(__file__).parent / "logs"


def _ensure_colab_stub() -> None:
    """Provide a lightweight google.colab stub when running outside Colab."""
    try:
        if importlib.util.find_spec("google.colab") is not None:  # pragma: no cover
            return
    except Exception:
        # If importlib itself fails, proceed to create the stub.
        pass

    if "google" not in sys.modules:
        google_mod = types.ModuleType("google")
        sys.modules["google"] = google_mod
    else:
        google_mod = sys.modules["google"]

    colab_mod = getattr(google_mod, "colab", None)
    if colab_mod is None:
        colab_mod = types.ModuleType("google.colab")
        google_mod.colab = colab_mod
        sys.modules["google.colab"] = colab_mod

    if not hasattr(colab_mod, "files"):
        class _Files:  # type: ignore
            def upload(self):
                raise RuntimeError(
                    "google.colab.files.upload() is unavailable outside Google Colab. "
                    "Place required input files locally and adjust the code accordingly."
                )

        colab_mod.files = _Files()


def prepare_logging(run_id: str) -> Path:
    """Configure logging to file and console using original stdout."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"hm_taha_run_{run_id}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info("Logging initialized. Writing to %s", log_path)
    return log_path


def run_modules() -> None:
    """Execute each module sequentially, capturing stdout/stderr per cell."""
    _ensure_colab_stub()

    if not MODULES_DIR.exists():
        raise FileNotFoundError(f"Modules directory not found: {MODULES_DIR}")

    module_paths = sorted(MODULES_DIR.glob("cell*.py"))
    if not module_paths:
        raise FileNotFoundError(f"No modules found in {MODULES_DIR}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = prepare_logging(run_id)

    logging.info("Starting HM_Taha module run (%s modules)", len(module_paths))
    logging.info("Original notebook outputs stored in %s", ORIGINAL_OUTPUTS_DIR)

    run_outputs_dir = RUNTIME_OUTPUTS_ROOT / run_id
    run_outputs_dir.mkdir(parents=True, exist_ok=True)

    shared_namespace = {"__name__": "__main__"}
    results = []

    for module_path in module_paths:
        cell_name = module_path.stem
        logging.info("Running %s", cell_name)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        status = "success"
        error_message = None

        original_input = builtins.input

        def _input_stub(prompt: str = "") -> str:
            raise RuntimeError(
                f"Interactive input requested during {cell_name}: {prompt!r}. "
                "Automated driver runs do not support stdin prompts."
            )

        builtins.input = _input_stub
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                try:
                    runpy.run_path(str(module_path), init_globals=shared_namespace, run_name="__main__")
                except BaseException as exc:  # pragma: no cover  # pylint: disable=broad-except
                    status = "error"
                    error_message = str(exc)
                    logging.exception("Error while executing %s", cell_name)
        finally:
            builtins.input = original_input

        output_text = stdout_buffer.getvalue()
        err_text = stderr_buffer.getvalue()
        combined_output = output_text
        if err_text:
            combined_output += '\n[stderr]\n' + err_text

        output_path = run_outputs_dir / f"{cell_name}_output.txt"
        output_path.write_text(combined_output, encoding="utf-8")
        logging.info(
            "%s completed with status=%s; output written to %s (%d bytes)",
            cell_name,
            status,
            output_path,
            output_path.stat().st_size,
        )

        results.append({
            "cell": cell_name,
            "status": status,
            "error": error_message,
            "output_path": output_path,
        })

    failures = [r for r in results if r["status"] != "success"]
    if failures:
        logging.warning("Run finished with %d failures.", len(failures))
        for item in failures:
            logging.warning(" - %s failed: %s", item["cell"], item["error"])
    else:
        logging.info("Run completed successfully with no errors.")

    logging.info("Runtime outputs saved under %s", run_outputs_dir)
    logging.info("Log written to %s", log_path)


if __name__ == "__main__":
    run_modules()
