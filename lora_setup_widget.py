"""UI helpers for configuring the XL LoRA trainer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import ipywidgets as widgets

__all__ = ["render_setup", "get_params"]

_PROJECTS_ROOT = Path("/teamspace/studios/this_studio/lora_projects")
_DEFAULT_DATASET_ROOT = Path("/teamspace/studios/this_studio/datasets")
_DEFAULT_OUTPUT_ROOT = Path("/teamspace/studios/this_studio/output")

_training_model_options = [
    "Pony Diffusion V6 XL",
    "Animagine XL V3",
    "animagine_4.0_zero",
    "Illustrious_0.1",
    "Illustrious_2.0",
    "NoobAI-XL0.75",
    "NoobAI-XL0.5",
    "Stable Diffusion XL 1.0 base",
    "NoobAIXL0_75vpred",
    "RouWei_v080vpred",
]
_training_model_default = "Illustrious_2.0"

_force_load_diffusers_options = [("False", False), ("True", True)]

_lr_scheduler_options = [
    "constant",
    "cosine",
    "cosine_with_restarts",
    "constant_with_warmup",
    "rex",
]
_lr_scheduler_default = "constant_with_warmup"

_lora_type_options = ["LoRA", "LoCon"]

_precision_options = ["full fp16", "full bf16", "mixed fp16", "mixed bf16"]

_optimizer_options = [
    "AdamW8bit",
    "Prodigy",
    "DAdaptation",
    "DadaptAdam",
    "DadaptLion",
    "AdamW",
    "AdaFactor",
    "Came",
]


@dataclass
class _State:
    ui: widgets.Widget | None = None
    status: widgets.HTML = widgets.HTML()
    steps: widgets.HTML = widgets.HTML()
    directory_hint: widgets.HTML = widgets.HTML()
    widgets: Dict[str, widgets.Widget] = None  # type: ignore[assignment]
    params: Dict[str, object] = None  # type: ignore[assignment]


_STATE = _State(widgets={}, params={})


def _coalesce(value, options, fallback):
    return value if value in options else fallback


def _format_scientific(value):
    try:
        return format(float(value), ".0e")
    except (TypeError, ValueError):
        return str(value)


def _ensure_directories(project_name: str) -> tuple[Path, Path]:
    if not project_name:
        dataset = _DEFAULT_DATASET_ROOT
        output = _DEFAULT_OUTPUT_ROOT
    else:
        project_root = _PROJECTS_ROOT / project_name
        dataset = project_root / "dataset"
        output = project_root / "output"
    dataset.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    return dataset, output


def _update_directory_hint(*_):
    project_name = _STATE.widgets["project_name"].value.strip()
    if project_name:
        base = _PROJECTS_ROOT / project_name
        dataset = base / "dataset"
        output = base / "output"
    else:
        dataset = _DEFAULT_DATASET_ROOT
        output = _DEFAULT_OUTPUT_ROOT
    _STATE.directory_hint.value = (
        f"<small><b>dataset:</b> <code>{dataset}</code><br>"
        f"<b>output:</b> <code>{output}</code></small>"
    )
    _STATE.params["dataset_dir"] = str(dataset)
    _STATE.params["output_dir"] = str(output)


def _update_steps(*_):
    try:
        num_images = int(_STATE.widgets["num_images"].value)
        repeats = int(_STATE.widgets["num_repeats"].value)
        epochs = int(_STATE.widgets["how_many"].value)
        batch_size = int(_STATE.widgets["train_batch_size"].value)
    except Exception:
        _STATE.steps.value = ""
        return
    if batch_size <= 0:
        _STATE.steps.value = ""
        return
    steps = (num_images * repeats * epochs) // batch_size
    _STATE.steps.value = f"<b>Pasos estimados:</b> {steps:,}".replace(",", "·")
    _STATE.params["steps"] = steps


def _parse_float(text: str, fallback: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return fallback


def _apply_params(*_):
    project_name = _STATE.widgets["project_name"].value.strip()
    dataset, output = _ensure_directories(project_name)

    params = {
        "project_name": project_name or "lora_project",
        "training_model": _STATE.widgets["training_model"].value,
        "force_load_diffusers": bool(_STATE.widgets["force_load_diffusers"].value),
        "resolution": int(_STATE.widgets["resolution"].value),
        "num_repeats": int(_STATE.widgets["num_repeats"].value),
        "how_many": int(_STATE.widgets["how_many"].value),
        "unet_lr": _parse_float(_STATE.widgets["unet_lr"].value, 1e-4),
        "text_encoder_lr": _parse_float(_STATE.widgets["text_encoder_lr"].value, 5e-5),
        "lr_scheduler": _STATE.widgets["lr_scheduler"].value,
        "lora_type": _STATE.widgets["lora_type"].value,
        "network_dim": int(_STATE.widgets["network_dim"].value),
        "network_alpha": int(_STATE.widgets["network_alpha"].value),
        "train_batch_size": int(_STATE.widgets["train_batch_size"].value),
        "precision": _STATE.widgets["precision"].value,
        "optimizer": _STATE.widgets["optimizer"].value,
        "optional_custom_training_model": _STATE.widgets["model_path"].value.strip(),
        "num_images": int(_STATE.widgets["num_images"].value),
    }
    params["dataset_dir"] = str(dataset)
    params["output_dir"] = str(output)
    params["repeats"] = params["num_repeats"]
    params["epochs"] = params["how_many"]
    params["batch_size"] = params["train_batch_size"]
    params["model_path"] = params["optional_custom_training_model"]
    _STATE.params.update(params)

    _update_directory_hint()

    _STATE.widgets["unet_lr"].value = _format_scientific(params["unet_lr"])
    _STATE.widgets["text_encoder_lr"].value = _format_scientific(params["text_encoder_lr"])

    _STATE.status.value = "<b>Parámetros actualizados.</b>"
    _update_steps()


def render_setup() -> widgets.Widget:
    if _STATE.ui is not None:
        return _STATE.ui

    project_name = widgets.Text(
        value=str(globals().get("project_name", "")),
        description="project_name",
        placeholder="Nombre del proyecto",
    )
    training_model = widgets.Dropdown(
        options=_training_model_options,
        value=_coalesce(globals().get("training_model", _training_model_default), _training_model_options, _training_model_default),
        description="training_model",
    )
    force_load_diffusers = widgets.Dropdown(
        options=_force_load_diffusers_options,
        value=bool(globals().get("force_load_diffusers", False)),
        description="force_load_diffusers",
    )
    resolution = widgets.IntText(
        value=int(globals().get("resolution", 1024)),
        description="resolution",
    )
    num_repeats = widgets.IntText(
        value=int(globals().get("num_repeats", 2)),
        description="num_repeats",
    )
    how_many = widgets.IntText(
        value=int(globals().get("how_many", 40)),
        description="how_many",
    )
    num_images = widgets.IntText(
        value=int(globals().get("num_images", 40)),
        description="num_images",
    )
    unet_lr = widgets.Text(
        value=_format_scientific(globals().get("unet_lr", 1e-4)),
        description="unet_lr",
    )
    text_encoder_lr = widgets.Text(
        value=_format_scientific(globals().get("text_encoder_lr", 5e-5)),
        description="text_encoder_lr",
    )
    lr_scheduler = widgets.Dropdown(
        options=_lr_scheduler_options,
        value=_coalesce(globals().get("lr_scheduler", _lr_scheduler_default), _lr_scheduler_options, _lr_scheduler_default),
        description="lr_scheduler",
    )
    lora_type = widgets.Dropdown(
        options=_lora_type_options,
        value=_coalesce(globals().get("lora_type", _lora_type_options[0]), _lora_type_options, _lora_type_options[0]),
        description="lora_type",
    )
    network_dim = widgets.IntText(
        value=int(globals().get("network_dim", 16)),
        description="network_dim",
    )
    network_alpha = widgets.IntText(
        value=int(globals().get("network_alpha", 32)),
        description="network_alpha",
    )
    train_batch_size = widgets.IntText(
        value=int(globals().get("train_batch_size", 8)),
        description="train_batch_size",
    )

    precision_default = globals().get("precision", "mixed fp16")
    if precision_default not in _precision_options:
        if "bf16" in str(precision_default):
            precision_default = "mixed bf16"
        elif "fp16" in str(precision_default):
            precision_default = "mixed fp16"
        else:
            precision_default = _precision_options[0]

    precision = widgets.Dropdown(
        options=_precision_options,
        value=precision_default,
        description="precision",
    )

    optimizer_default = globals().get("optimizer", "Prodigy")
    if optimizer_default not in _optimizer_options:
        optimizer_default = "Prodigy"

    optimizer = widgets.Dropdown(
        options=_optimizer_options,
        value=optimizer_default,
        description="optimizer",
    )

    model_path = widgets.Text(
        value=str(globals().get("optional_custom_training_model", "")),
        description="model_path",
        placeholder="URL o ruta local (opcional)",
    )

    apply_button = widgets.Button(
        description="Aplicar parámetros",
        button_style="success",
        icon="check",
        layout=widgets.Layout(width="auto"),
    )
    apply_button.on_click(_apply_params)

    for name, widget in {
        "project_name": project_name,
        "training_model": training_model,
        "force_load_diffusers": force_load_diffusers,
        "resolution": resolution,
        "num_repeats": num_repeats,
        "how_many": how_many,
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "lr_scheduler": lr_scheduler,
        "lora_type": lora_type,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "train_batch_size": train_batch_size,
        "precision": precision,
        "optimizer": optimizer,
        "model_path": model_path,
        "num_images": num_images,
    }.items():
        _STATE.widgets[name] = widget

    for field in (num_images, num_repeats, how_many, train_batch_size):
        field.observe(_update_steps, "value")

    project_name.observe(_update_directory_hint, "value")

    grid = widgets.GridBox(
        [
            project_name,
            training_model,
            force_load_diffusers,
            model_path,
            resolution,
            num_repeats,
            how_many,
            num_images,
            unet_lr,
            text_encoder_lr,
            lr_scheduler,
            lora_type,
            network_dim,
            network_alpha,
            train_batch_size,
            precision,
            optimizer,
        ],
        layout=widgets.Layout(
            grid_template_columns="repeat(2, minmax(0, 1fr))",
            grid_gap="12px",
            width="100%",
        ),
    )

    container = widgets.VBox(
        [
            widgets.HTML("<h2>🔧 Configuración rápida del entrenamiento</h2>"),
            grid,
            _STATE.steps,
            _STATE.directory_hint,
            widgets.HBox([apply_button]),
            _STATE.status,
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )

    _STATE.ui = container
    _apply_params()
    return container


def get_params() -> Dict[str, object]:
    if not _STATE.params:
        raise RuntimeError("render_setup() debe llamarse antes de get_params().")
    return dict(_STATE.params)
