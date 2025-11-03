"""UI helpers for configuring the Kohya training session from the notebook.

The original notebook embedded all of the ipywidgets setup logic directly in a
cell.  Moving the implementation to this module keeps the notebook tidy while
preserving the same behaviour.  The only requirement from the notebook is to
call :func:`render_quick_training_config` passing ``globals()`` so the widgets
can read and update the shared state exactly like before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import ipywidgets as widgets
from IPython.display import Markdown, display
from pathlib import Path


PROJECTS_ROOT = Path("/teamspace/studios/this_studio/lora_projects")


@dataclass(frozen=True)
class _DropdownOption:
    label: str
    value: Any

    def as_tuple(self) -> Tuple[str, Any]:
        return self.label, self.value


def _format_scientific(value: Any) -> str:
    """Return ``value`` formatted in scientific notation when possible."""
    try:
        return format(float(value), ".0e")
    except (TypeError, ValueError):
        return str(value)


def _resolve_precision(default: str, options: Iterable[str]) -> str:
    if default in options:
        return default
    default = str(default)
    if "bf16" in default:
        return "mixed bf16"
    if "fp16" in default:
        return "mixed fp16"
    return next(iter(options))


def render_quick_training_config(namespace: Dict[str, Any]) -> None:
    """Render the interactive configuration UI.

    Parameters
    ----------
    namespace:
        Usually the ``globals()`` dictionary from the notebook.  The widgets
        will read default values from it and write the updated selections back
        into the same mapping so that the following cells behave exactly like
        the original implementation.
    """

    training_model_options = [
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

    force_load_diffusers_options = [
        _DropdownOption("❌ No (ckpt)", False),
        _DropdownOption("✅ Sí (diffusers)", True),
    ]

    lr_scheduler_options = [
        "constant",
        "cosine",
        "cosine_with_restarts",
        "constant_with_warmup",
        "rex",
    ]

    precision_options = [
        "full fp16",
        "full bf16",
        "mixed fp16",
        "mixed bf16",
    ]

    optimizer_options = [
        "AdamW8bit",
        "Prodigy",
        "DAdaptation",
        "DadaptAdam",
        "DadaptLion",
        "AdamW",
        "AdaFactor",
        "Came",
    ]

    # Widgets -----------------------------------------------------------------
    base_style = {"description_width": "160px"}
    number_layout = widgets.Layout(width="100%")

    project_name_widget = widgets.Text(
        value=str(namespace.get("project_name", "")),
        description="project_name",
        placeholder="Nombre del proyecto",
        style=base_style,
    )
    training_model_widget = widgets.Dropdown(
        options=training_model_options,
        value=namespace.get("training_model", "Illustrious_2.0"),
        description="training_model",
        style=base_style,
    )
    force_load_diffusers_widget = widgets.Dropdown(
        options=[opt.as_tuple() for opt in force_load_diffusers_options],
        value=bool(namespace.get("force_load_diffusers", False)),
        description="force_load_diffusers",
        style=base_style,
    )
    resolution_widget = widgets.IntText(
        value=int(namespace.get("resolution", 1024)),
        description="resolution",
        style=base_style,
        layout=number_layout,
    )
    num_repeats_widget = widgets.IntText(
        value=int(namespace.get("num_repeats", 2)),
        description="num_repeats",
        style=base_style,
        layout=number_layout,
    )
    how_many_widget = widgets.IntText(
        value=int(namespace.get("how_many", 40)),
        description="how_many",
        style=base_style,
        layout=number_layout,
    )
    unet_lr_widget = widgets.Text(
        value=_format_scientific(namespace.get("unet_lr", 1e-4)),
        description="unet_lr",
        style=base_style,
    )
    text_encoder_lr_widget = widgets.Text(
        value=_format_scientific(namespace.get("text_encoder_lr", 5e-5)),
        description="text_encoder_lr",
        style=base_style,
    )
    lr_scheduler_widget = widgets.Dropdown(
        options=lr_scheduler_options,
        value=namespace.get("lr_scheduler", "constant_with_warmup"),
        description="lr_scheduler",
        style=base_style,
    )
    lora_type_widget = widgets.Dropdown(
        options=["LoRA", "LoCon"],
        value=namespace.get("lora_type", "LoRA"),
        description="lora_type",
        style=base_style,
    )
    network_dim_widget = widgets.IntText(
        value=int(namespace.get("network_dim", 16)),
        description="network_dim",
        style=base_style,
        layout=number_layout,
    )
    network_alpha_widget = widgets.IntText(
        value=int(namespace.get("network_alpha", 32)),
        description="network_alpha",
        style=base_style,
        layout=number_layout,
    )
    train_batch_size_widget = widgets.IntText(
        value=int(namespace.get("train_batch_size", 8)),
        description="train_batch_size",
        style=base_style,
        layout=number_layout,
    )
    precision_widget = widgets.Dropdown(
        options=precision_options,
        value=_resolve_precision(namespace.get("precision", "fp16"), precision_options),
        description="precision",
        style=base_style,
    )
    optimizer_widget = widgets.Dropdown(
        options=optimizer_options,
        value=namespace.get("optimizer", "Prodigy"),
        description="optimizer",
        style=base_style,
    )

    status_output = widgets.HTML()
    apply_button = widgets.Button(
        description="Aplicar parámetros",
        button_style="success",
        icon="check",
        layout=widgets.Layout(width="auto", align_self="flex-end"),
    )

    grid_layout = widgets.Layout(
        grid_template_columns="repeat(2, minmax(0, 1fr))",
        grid_gap="12px",
        width="100%",
    )
    basics_grid = widgets.GridBox(
        children=[
            project_name_widget,
            training_model_widget,
            force_load_diffusers_widget,
            resolution_widget,
            num_repeats_widget,
            how_many_widget,
        ],
        layout=grid_layout,
    )
    advanced_grid = widgets.GridBox(
        children=[
            unet_lr_widget,
            text_encoder_lr_widget,
            lr_scheduler_widget,
            lora_type_widget,
            network_dim_widget,
            network_alpha_widget,
            train_batch_size_widget,
            precision_widget,
            optimizer_widget,
        ],
        layout=grid_layout,
    )

    def apply_params(_=None) -> None:
        try:
            updates = {
                "project_name": project_name_widget.value.strip(),
                "training_model": training_model_widget.value,
                "force_load_diffusers": bool(force_load_diffusers_widget.value),
                "resolution": int(resolution_widget.value),
                "num_repeats": int(num_repeats_widget.value),
                "how_many": int(how_many_widget.value),
                "unet_lr": float(unet_lr_widget.value),
                "text_encoder_lr": float(text_encoder_lr_widget.value),
                "lr_scheduler": lr_scheduler_widget.value,
                "lora_type": lora_type_widget.value,
                "network_dim": int(network_dim_widget.value),
                "network_alpha": int(network_alpha_widget.value),
                "train_batch_size": int(train_batch_size_widget.value),
                "precision": precision_widget.value,
                "optimizer": optimizer_widget.value,
            }
        except ValueError as exc:
            status_output.value = f"<b>Error:</b> {exc}"
            return

        namespace.update(updates)
        unet_lr_widget.value = _format_scientific(updates["unet_lr"])
        text_encoder_lr_widget.value = _format_scientific(updates["text_encoder_lr"])

        project_name_value = updates["project_name"]
        directory_message = ""
        if project_name_value:
            project_dir = PROJECTS_ROOT / project_name_value
            dataset_dir = project_dir / "dataset"
            try:
                dataset_dir.mkdir(parents=True, exist_ok=True)
                directory_message = f" Directorios preparados en <code>{project_dir}</code>."
            except Exception as exc:  # pragma: no cover - user environment errors
                status_output.value = f"<b>Error al preparar directorios:</b> {exc}"
                return

        status_output.value = f"<b>Parámetros actualizados correctamente.</b>{directory_message}"

    apply_button.on_click(apply_params)

    display(Markdown("""
### Configuración rápida del entrenamiento
Personaliza tu sesión desde este panel compacto. Haz clic en **Aplicar parámetros** para guardar los cambios.
"""))
    display(
        widgets.VBox(
            [
                widgets.HTML("<h4 style='margin-bottom:4px;'>Datos básicos</h4>"),
                basics_grid,
                widgets.HTML("<h4 style='margin:16px 0 4px;'>Ajustes avanzados</h4>"),
                advanced_grid,
                widgets.HBox([widgets.HBox([], layout=widgets.Layout(flex="1")), apply_button]),
                status_output,
            ]
        )
    )

    apply_params()


__all__ = ["render_quick_training_config"]
