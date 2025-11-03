"""Training runner for the XL LoRA notebook."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import ipywidgets as widgets
from IPython.display import Markdown, display
from huggingface_hub.utils import disable_progress_bars
from tqdm.auto import tqdm, trange  # noqa: F401
import toml

__all__ = ["run_training"]

ROOT_DIR = Path("/teamspace/studios/this_studio")
TRAINER_DIR = ROOT_DIR / "LoRA_Easy_Training_scripts_Backend"
KOHYA_DIR = TRAINER_DIR / "sd-scripts"
MODELS_DIR = ROOT_DIR / "models"
DOWNLOADS_DIR = ROOT_DIR / "downloads"
CUSTOM_OPTIMIZER_PATH = TRAINER_DIR / "custom_scheduler"
PYTHON_BIN = Path("/home/zeus/miniconda3/envs/cloudspace/bin/python3")
TRAIN_NETWORK = KOHYA_DIR / "sdxl_train_network.py"

COMMIT = "fa2427c6b468231e8e270e40fe72add780118dbe"
LOWRAM = False
LOAD_TRUNCATED_IMAGES = True
BETTER_EPOCH_NAMES = True
FIX_DIFFUSERS = True
FIX_WANDB_WARNING = True

SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _configure_logging():
    logging.getLogger().handlers = [logging.StreamHandler()]
    logging.getLogger().setLevel(logging.INFO)


def _prepare_environment():
    os.environ.setdefault("TQDM_MININTERVAL", "2")
    os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "1")
    sys.path.insert(0, str(CUSTOM_OPTIMIZER_PATH))
    os.environ["PYTHONPATH"] = str(CUSTOM_OPTIMIZER_PATH) + os.pathsep + os.environ.get("PYTHONPATH", "")
    disable_progress_bars()


def _run_command(command: Iterable[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    cmd_list = list(command)
    logging.info("$ %s", " ".join(cmd_list))
    process = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Command {' '.join(cmd_list)} failed with exit code {ret}")


def _replace_in_file(path: Path, pattern: str, replacement: str) -> None:
    if not path.exists():
        return
    content = path.read_text()
    if pattern not in content:
        return
    path.write_text(content.replace(pattern, replacement))


def _install_trainer():
    libtcmalloc_path = ROOT_DIR / "libtcmalloc_minimal.so.4"

    if not libtcmalloc_path.exists():
        _run_command(
            [
                "wget",
                "-q",
                "-c",
                "--show-progress",
                "https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4",
                "-O",
                str(libtcmalloc_path),
            ]
        )

    if not TRAINER_DIR.exists():
        _run_command(["git", "clone", "-b", "dev", "https://github.com/gwhitez/LoRA_Easy_Training_scripts_Backend.git", str(TRAINER_DIR)])
    else:
        _run_command(["git", "pull"], cwd=TRAINER_DIR)

    install_script = TRAINER_DIR / "colab_install.sh"
    if install_script.exists():
        install_script.chmod(0o755)
        installer_py = TRAINER_DIR / "installer.py"
        _replace_in_file(
            installer_py,
            "uv venv venv --python 3.10.16",
            "uv venv venv --python 3.10.16 --link-mode copy",
        )
        venv_dir = TRAINER_DIR / "venv"
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "UV_LINK_MODE": "copy"}
        _run_command([str(install_script)], cwd=TRAINER_DIR, env=env)

    if LOAD_TRUNCATED_IMAGES:
        target = KOHYA_DIR / "library" / "train_util.py"
        content = target.read_text()
        replacement = "from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True"
        content = re.sub(r"from PIL import Image", replacement, content, count=1)
        target.write_text(content)
    if BETTER_EPOCH_NAMES:
        train_util = KOHYA_DIR / "library" / "train_util.py"
        train_util.write_text(train_util.read_text().replace("{:06d}", "{:02d}"))
        train_network = KOHYA_DIR / "train_network.py"
        train_network.write_text(
            train_network.read_text().replace("\".\" + args.save_model_as)", '"-{:02d}.".format(num_train_epochs) + args.save_model_as)')
        )
    if FIX_DIFFUSERS:
        deprecation_utils = Path("/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/diffusers/utils/deprecation_utils.py")
        _replace_in_file(deprecation_utils, "if version.parse", "if False:#")
    if FIX_WANDB_WARNING:
        for fname in ("train_network.py", "sdxl_train.py"):
            path = KOHYA_DIR / fname
            _replace_in_file(path, "accelerator.log(logs, step=epoch + 1)", "")

    os.environ["LD_PRELOAD"] = str(libtcmalloc_path)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"


def _validate_dataset(dataset_dir: Path, repeats: int) -> Dict[str, int]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"La carpeta {dataset_dir} no existe")

    counts: Dict[str, int] = {}
    for entry in dataset_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_TYPES:
            counts.setdefault(str(dataset_dir), 0)
            counts[str(dataset_dir)] += 1
    image_count = counts.get(str(dataset_dir), 0)
    if image_count == 0:
        raise ValueError("La carpeta del dataset está vacía")
    logging.info("📈 Se encontró %s imágenes con %s repeticiones", image_count, repeats)
    return counts


def _download_model(model_url: str, model_file: Path, vae_file: Path, vae_url: Optional[str]) -> None:
    if re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", model_url):
        model_url = model_url.replace("blob", "resolve")

    model_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info("🌐 Descargando modelo en %s", model_file)
    _run_command(
        [
            "aria2c",
            model_url,
            "--console-log-level=warn",
            "-c",
            "-s",
            "16",
            "-x",
            "16",
            "-k",
            "10M",
            "-d",
            str(model_file.parent),
            "-o",
            model_file.name,
        ]
    )

    if vae_url and not vae_file.exists():
        logging.info("🌐 Descargando VAE en %s", vae_file)
        _run_command(
            [
                "aria2c",
                vae_url,
                "--console-log-level=warn",
                "-c",
                "-s",
                "16",
                "-x",
                "16",
                "-k",
                "10M",
                "-d",
                str(vae_file.parent),
                "-o",
                vae_file.name,
            ]
        )


def _create_config(params: Dict[str, object], model_file: Path, vae_file: Path, output_dir: Path, config_dir: Path) -> tuple[Path, Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_config_file = config_dir / "dataset_config.toml"
    training_config_file = config_dir / "training_config.toml"

    num_repeats = int(params.get("num_repeats", 2))
    caption_extension = params.get("caption_extension", ".txt")
    dataset_dir = Path(params["dataset_dir"])

    dataset_config = {
        "datasets": [
            {
                "resolution": int(params.get("resolution", 1024)),
                "min_bucket_reso": 256,
                "max_bucket_reso": 4096,
                "bucket_reso_steps": 64,
                "caption_extension": caption_extension,
                "flip_aug": bool(params.get("flip_aug", False)),
                "shuffle_caption": bool(params.get("shuffle_caption", True)),
                "keep_tokens": int(params.get("keep_tokens", 0)),
                "color_aug": False,
                "debug_dataset": False,
                "subsets": [
                    {
                        "image_dir": str(dataset_dir),
                        "num_repeats": num_repeats,
                        "caption_dropout_every_n_epochs": 0,
                        "caption_dropout_rate": 0,
                        "shuffle_caption": bool(params.get("shuffle_caption", True)),
                        "keep_tokens": int(params.get("keep_tokens", 0)),
                    }
                ],
            }
        ]
    }

    with dataset_config_file.open("w", encoding="utf-8") as f:
        toml.dump(dataset_config, f)

    training_config = {
        "network_arguments": {
            "unet_lr": float(params.get("unet_lr", 1e-4)),
            "text_encoder_lr": float(params.get("text_encoder_lr", 5e-5)),
            "network_dim": int(params.get("network_dim", 16)),
            "network_alpha": int(params.get("network_alpha", 32)),
            "network_module": "networks.lora",
            "network_train_unet_only": False,
        },
        "optimizer_arguments": {
            "learning_rate": float(params.get("unet_lr", 1e-4)),
            "lr_scheduler": params.get("lr_scheduler", "constant_with_warmup"),
            "lr_warmup_steps": int(params.get("lr_warmup_steps", 100)),
            "optimizer_type": params.get("optimizer", "Prodigy"),
            "optimizer_args": params.get("optimizer_args", []),
            "loss_type": "l2",
            "max_grad_norm": 1.0,
        },
        "training_arguments": {
            "lowram": LOWRAM,
            "pretrained_model_name_or_path": str(model_file),
            "vae": str(vae_file),
            "max_train_steps": int(params.get("max_train_steps", 0)) or None,
            "max_train_epochs": int(params.get("how_many", 10)),
            "train_batch_size": int(params.get("train_batch_size", 8)),
            "seed": int(params.get("seed", 42)),
            "xformers": True,
            "sdpa": False,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps", 1)),
            "mixed_precision": params.get("precision", "mixed fp16"),
            "cache_latents": bool(params.get("cache_latents", True)),
            "cache_latents_to_disk": bool(params.get("cache_latents_to_disk", False)),
            "cache_text_encoder_outputs": bool(params.get("cache_text_encoder_outputs", False)),
            "output_dir": str(output_dir),
            "logging_dir": str(output_dir / "logs"),
            "save_every_n_epochs": int(params.get("save_every_n_epochs", 1)),
            "keep_only_last_n_epochs": int(params.get("keep_only_last_n_epochs", 5)),
        },
    }

    with training_config_file.open("w", encoding="utf-8") as f:
        toml.dump(training_config, f)

    return training_config_file, dataset_config_file


def _resolve_model(params: Dict[str, object]) -> tuple[str, Path]:
    model_url = str(params.get("model_path") or params.get("optional_custom_training_model") or "").strip()
    training_model = str(params.get("training_model", "Illustrious_2.0"))
    force_diffusers = bool(params.get("force_load_diffusers", False))

    if not model_url:
        model_map = {
            "Pony Diffusion V6 XL": (
                "https://huggingface.co/WhiteAiZ/PonyXL/resolve/main/PonyDiffusionV6XL.safetensors",
                "ponyDiffusionV6XL.safetensors",
            ),
            "Animagine XL V3": (
                "https://civitai.com/api/download/models/293564",
                "animagineXLV3.safetensors",
            ),
            "animagine_4.0_zero": (
                "https://huggingface.co/cagliostrolab/animagine-xl-4.0-zero/resolve/main/animagine-xl-4.0-zero.safetensors",
                "animagine-xl-4.0-zero.safetensors",
            ),
            "Illustrious_0.1": (
                "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors",
                "Illustrious-XL-v0.1.safetensors",
            ),
            "Illustrious_2.0": (
                "https://huggingface.co/WhiteAiZ/Illustrious_2.0/resolve/main/illustriousXL20_v20.safetensors",
                "illustriousXL20_v20.safetensors",
            ),
            "NoobAI-XL0.75": (
                "https://huggingface.co/Laxhar/noobai-XL-0.75/resolve/main/NoobAI-XL-v0.75.safetensors",
                "NoobAI-XL-v0.75.safetensors",
            ),
            "NoobAI-XL0.5": (
                "https://huggingface.co/Laxhar/noobai-XL-0.5/resolve/main/NoobAI-XL-v0.5.safetensors",
                "NoobAI-XL-v0.5.safetensors",
            ),
            "Stable Diffusion XL 1.0 base": (
                "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                "sd_xl_base_1.0.safetensors",
            ),
            "NoobAIXL0_75vpred": (
                "https://huggingface.co/Laxhar/noobai-XL-Vpred-0.75/resolve/main/NoobAI-XL-Vpred-v0.75.safetensors",
                "NoobAI-XL-Vpred-v0.75.safetensors",
            ),
            "RouWei_v080vpred": (
                "https://huggingface.co/WhiteAiZ/RouWei/resolve/main/rouwei_v080Vpred.safetensors",
                "rouwei_v080Vpred.safetensors",
            ),
        }
        model_url, filename = model_map.get(training_model, ("", "model.safetensors"))
        if force_diffusers and "huggingface.co" in model_url:
            model_url = re.sub(r"/resolve/main/.*", "", model_url)
        if not model_url:
            raise ValueError("No se pudo determinar la URL del modelo base.")
    else:
        filename = Path(model_url).name or "model.safetensors"

    model_file = MODELS_DIR / filename
    return model_url, model_file


def run_training(params: Dict[str, object]) -> widgets.Output:
    output = widgets.Output()
    _configure_logging()
    _prepare_environment()

    for directory in (MODELS_DIR, DOWNLOADS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    with output:
        logging.info("🔵 Lightning environment detectado. Detén el cuaderno manualmente cuando termines.")

        dataset_dir = Path(params["dataset_dir"])
        output_dir = Path(params["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        num_images = int(params.get("num_images", 0))
        repeats = int(params.get("num_repeats", params.get("repeats", 1)))
        epochs = int(params.get("how_many", params.get("epochs", 1)))
        batch_size = int(params.get("train_batch_size", params.get("batch_size", 1)))
        if batch_size <= 0:
            raise ValueError("train_batch_size debe ser mayor a 0")
        steps = (num_images * repeats * epochs) // batch_size
        params.setdefault("max_train_steps", steps)
        logging.info("Pasos estimados: %s", steps)

        _validate_dataset(dataset_dir, repeats)

        if not params.get("dependencies_installed"):
            logging.info("🏭 Instalando entrenador...")
            t0 = time.time()
            _install_trainer()
            params["dependencies_installed"] = True
            logging.info("✅ Instalación terminada en %s segundos.", int(time.time() - t0))
        else:
            logging.info("✅ Dependencias ya instaladas.")

        model_url, model_file = _resolve_model(params)
        vae_file = MODELS_DIR / "sdxl_vae.safetensors"
        vae_url = "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"

        if model_url.startswith("http"):
            _download_model(model_url, model_file, vae_file, vae_url)
        else:
            model_path = Path(model_url)
            if model_path.exists():
                model_file = model_path
            elif not model_file.exists():
                raise FileNotFoundError(f"No se encontró el modelo en {model_url}")

        config_dir = output_dir / "config"
        training_config_file, dataset_config_file = _create_config(params, model_file, vae_file, output_dir, config_dir)

        logging.info("⭐ Iniciando Entrenador..")
        env = os.environ.copy()
        env.update({"HF_HOME": str(ROOT_DIR / ".cache")})
        _run_command(
            [
                str(PYTHON_BIN),
                str(TRAIN_NETWORK),
                f"--config_file={training_config_file}",
                f"--dataset_config={dataset_config_file}",
            ],
            cwd=KOHYA_DIR,
            env=env,
        )

        display(Markdown(f"### ✅ ¡Hecho! Tus archivos se encuentran en `{output_dir}`"))
        logging.info("🔵 El cuaderno continuará en ejecución. Detén manualmente la sesión de Lightning cuando termines.")

    return output
