"""Lightning LoRA training UI for JupyterLab widgets."""
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import ipywidgets as widgets
from IPython.display import Markdown, display
from huggingface_hub.utils import disable_progress_bars
from PIL import Image
from tqdm.auto import trange, tqdm
import toml


LIGHTNING_ROOT = Path("/teamspace/studios/this_studio")
TRAINER_REPO = "https://github.com/gwhitez/LoRA_Easy_Training_scripts_Backend.git"
TRAINER_BRANCH = "dev"
TRAINER_DIR = LIGHTNING_ROOT / "LoRA_Easy_Training_scripts_Backend"
KOHYA_DIR = TRAINER_DIR / "sd-scripts"
MODELS_DIR = LIGHTNING_ROOT / "models"
DOWNLOADS_DIR = LIGHTNING_ROOT / "downloads"
DEFAULT_PROJECTS_DIR = LIGHTNING_ROOT / "lora_projects"
DEFAULT_VENV_PYTHON = Path("/home/zeus/miniconda3/envs/cloudspace/bin/python3")
TRAIN_NETWORK = KOHYA_DIR / "sdxl_train_network.py"


@dataclass
class TrainerConfig:
    dataset_dir: Path = DEFAULT_PROJECTS_DIR / "example" / "dataset"
    output_dir: Path = DEFAULT_PROJECTS_DIR / "example" / "output"
    model_path: str = "https://huggingface.co/WhiteAiZ/Illustrious_2.0/resolve/main/illustriousXL20_v20.safetensors"
    epochs: int = 10
    repeats: int = 2
    batch_size: int = 8
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5
    num_images: int = 40
    precision: str = "mixed fp16"
    force_load_diffusers: bool = False
    custom_dataset: Optional[str] = None
    optional_custom_training_model: str = ""
    custom_model_is_diffusers: bool = False
    custom_model_is_vpred: bool = False
    wandb_key: str = ""
    folder_structure: str = "Organize by project (lora_projects/project_name/dataset)"
    project_name: str = "example"
    resolution: int = 1024
    flip_aug: bool = False
    caption_extension: str = ".txt"
    shuffle_tags: bool = True
    activation_tags: int = 1
    preferred_unit: str = "Epochs"
    how_many: int = 40
    save_every_n_epochs: int = 1
    keep_only_last_n_epochs: int = 5
    lr_scheduler: str = "constant_with_warmup"
    lr_scheduler_number: int = 0
    lr_warmup_ratio: float = 0.05
    lr_warmup_steps: int = 100
    min_snr_gamma_enabled: bool = True
    min_snr_gamma: float = 8.0
    ip_noise_gamma_enabled: bool = True
    ip_noise_gamma: float = 0.05
    multinoise: bool = False
    lora_type: str = "LoRA"
    network_dim: int = 16
    network_alpha: int = 32
    conv_dim: int = 16
    conv_alpha: int = 8
    cross_attention: str = "xformers"
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder_outputs: bool = False
    optimizer: str = "Prodigy"
    optimizer_args: List[str] = field(default_factory=list)
    recommended_values: bool = True
    custom_optimizer_path: Path = TRAINER_DIR / "custom_scheduler"
    gradient_accumulation_steps: int = 1
    bucket_reso_steps: int = 64
    min_bucket_reso: int = 256
    max_bucket_reso: int = 4096
    seed: int = 42
    max_token_length: int = 225
    multires_noise_iterations: Optional[int] = None
    multires_noise_discount: Optional[float] = None
    override_dataset_config_file: Optional[str] = None
    override_config_file: Optional[str] = None
    load_truncated_images: bool = True
    better_epoch_names: bool = True
    fix_diffusers: bool = True
    fix_wandb_warning: bool = True
    low_ram: bool = False
    lr_scheduler_type: Optional[str] = None
    lr_scheduler_args: Optional[List[str]] = None
    lr_scheduler_num_cycles: int = 0
    lr_scheduler_power: int = 0
    lr_warmup_steps_override: Optional[int] = None
    full_precision: bool = False
    wandb_enabled: bool = False

    def derive(self) -> None:
        # Normalise values derived from UI before training.
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)
        if not self.project_name:
            self.project_name = self.output_dir.name
        self.how_many = self.epochs if self.preferred_unit == "Epochs" else self.how_many


class TrainerApp:
    def __init__(self) -> None:
        self.config = TrainerConfig()
        self.dependencies_installed = False
        self.previous_model_url: Optional[str] = None
        self.model_file: Optional[Path] = None
        self.log_output = widgets.Output()
        disable_progress_bars(False)

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def configure_logging() -> None:
        os.environ.setdefault("TQDM_MININTERVAL", "2")
        os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "1")
        root_logger = logging.getLogger()
        root_logger.handlers = [logging.StreamHandler()]
        root_logger.setLevel(logging.INFO)

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def prepare_scheduler(self) -> str:
        cfg = self.config
        cfg.lr_scheduler_type = None
        existing_args = cfg.lr_scheduler_args[:] if cfg.lr_scheduler_args else []
        cfg.lr_scheduler_args = existing_args
        scheduler_name = cfg.lr_scheduler
        if "rex" in scheduler_name:
            scheduler_name = "cosine"
            cfg.lr_scheduler_type = (
                "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
            )
            base_args = ["min_lr=1e-9", "gamma=0.9", "d=0.9"]
            missing = [arg for arg in base_args if arg not in existing_args]
            cfg.lr_scheduler_args = missing + existing_args
        return scheduler_name

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> int:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        assert process.stdout is not None
        for line in process.stdout:
            with self.log_output:
                print(line, end="")
        return process.wait()

    def _resolve_model_url(self) -> str:
        config = self.config
        training_model = config.model_path.strip()
        return training_model

    # ------------------------------------------------------------------
    def install_trainer(self) -> None:
        with self.log_output:
            print("🏭 Instalando entrenador...")
        lib_path = LIGHTNING_ROOT / "libtcmalloc_minimal.so.4"
        if not lib_path.exists():
            self._run_command(
                [
                    "wget",
                    "-q",
                    "-c",
                    "--show-progress",
                    "https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4",
                    "-O",
                    str(lib_path),
                ]
            )
        if not TRAINER_DIR.exists():
            self._run_command(["git", "clone", "-b", TRAINER_BRANCH, TRAINER_REPO, str(TRAINER_DIR)])
        else:
            self._run_command(["git", "pull"], cwd=TRAINER_DIR)

        install_script = TRAINER_DIR / "colab_install.sh"
        if install_script.exists():
            install_script.chmod(0o755)
            self._run_command([str(install_script)], cwd=TRAINER_DIR)

        if self.config.load_truncated_images:
            self._patch_kohya_for_truncated_images()
        if self.config.better_epoch_names:
            self._patch_epoch_names()
        if self.config.fix_diffusers:
            self._patch_diffusers()
        if self.config.fix_wandb_warning:
            self._patch_wandb_logging()

        os.environ["LD_PRELOAD"] = str(lib_path)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        os.environ["SAFETENSORS_FAST_GPU"] = "1"
        os.environ["PYTHONWARNINGS"] = "ignore"
        self.dependencies_installed = True

    def _patch_kohya_for_truncated_images(self) -> None:
        target = KOHYA_DIR / "library" / "train_util.py"
        if not target.exists():
            return
        with open(target, "r", encoding="utf-8") as handle:
            content = handle.read()
        if "ImageFile.LOAD_TRUNCATED_IMAGES" not in content:
            content = content.replace(
                "from PIL import Image",
                "from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True",
            )
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(content)

    def _patch_epoch_names(self) -> None:
        mapping = {
            KOHYA_DIR / "library" / "train_util.py": [
                ("{:06d}", "{:02d}"),
            ],
            KOHYA_DIR / "train_network.py": [
                (
                    '"." + args.save_model_as)',
                    '"-{:02d}.".format(num_train_epochs) + args.save_model_as)',
                )
            ],
        }
        for file_path, replacements in mapping.items():
            if not file_path.exists():
                continue
            with open(file_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            updated = content
            for old, new in replacements:
                updated = updated.replace(old, new)
            if updated != content:
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

    def _patch_diffusers(self) -> None:
        utils_path = Path(
            "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/diffusers/utils/deprecation_utils.py"
        )
        if utils_path.exists():
            with open(utils_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            if "if False:#" not in content:
                updated = content.replace("if version.parse", "if False:#")
                with open(utils_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

    def _patch_wandb_logging(self) -> None:
        targets = [KOHYA_DIR / "train_network.py", KOHYA_DIR / "sdxl_train.py"]
        for path in targets:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
            updated = content.replace("accelerator.log(logs, step=epoch + 1)", "")
            if updated != content:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

    # ------------------------------------------------------------------
    def validate_dataset(self) -> bool:
        cfg = self.config
        supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        dataset_paths: Dict[Path, int] = {}
        if cfg.custom_dataset:
            try:
                parsed = toml.loads(cfg.custom_dataset)
            except Exception as exc:  # pragma: no cover - UI feedback path
                with self.log_output:
                    print(f"💥 Error: dataset TOML inválido ({exc}).")
                return False
            subsets = parsed["datasets"][0]["subsets"]
            for subset in subsets:
                folder = Path(subset["image_dir"])
                dataset_paths[folder] = int(subset["num_repeats"])
        else:
            dataset_paths[cfg.dataset_dir] = cfg.repeats

        missing = [path for path in dataset_paths if not path.exists()]
        if missing:
            with self.log_output:
                for path in missing:
                    print(f"💥 Error: La carpeta {path} no existe.")
            return False

        images_repeats: Dict[Path, tuple[int, int]] = {}
        for folder, repeats in dataset_paths.items():
            images = [f for f in folder.iterdir() if f.suffix.lower() in supported_types]
            images_repeats[folder] = (len(images), repeats)

        empty = [folder for folder, (count, _) in images_repeats.items() if count == 0]
        if empty:
            with self.log_output:
                for folder in empty:
                    print(f"💥 Error: la carpeta {folder} está vacía.")
            return False

        steps_per_epoch = 0
        with self.log_output:
            for folder, (img_count, repeats) in tqdm(
                images_repeats.items(), desc="Dataset", leave=False
            ):
                steps_per_epoch += img_count * repeats
                print(
                    f"📁 {folder} - {img_count} imágenes, repeticiones {repeats}, pasos {img_count * repeats}."
                )

        total_steps = steps_per_epoch / max(cfg.batch_size, 1)
        max_epochs = cfg.epochs
        approx_steps = int(total_steps * max_epochs)
        if approx_steps > 10000:
            with self.log_output:
                print("💥 Error: demasiados pasos. Revisa tus parámetros.")
            return False
        return True

    # ------------------------------------------------------------------
    def _model_destination(self, url: str) -> Path:
        suffix = Path(url.split("?")[0]).suffix or ".safetensors"
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(url).stem) + suffix
        return MODELS_DIR / filename

    def download_model(self) -> bool:
        cfg = self.config
        url = self._resolve_model_url()
        url = url.replace("blob", "resolve") if "huggingface" in url and "/blob/" in url else url
        candidate_path = Path(url)
        if candidate_path.exists():
            self.model_file = candidate_path
            self.previous_model_url = url
            return True

        self.model_file = self._model_destination(url)
        if self.model_file.exists():
            self.previous_model_url = url
            return True

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            "aria2c",
            url,
            "--console-log-level=warn",
            "-c",
            "-s",
            "16",
            "-x",
            "16",
            "-k",
            "10M",
            "-d",
            str(MODELS_DIR),
            "-o",
            self.model_file.name,
        ]
        with self.log_output:
            print(f"🌐 Descargando modelo en {self.model_file} ...")
        result = self._run_command(cmd)
        if result != 0:
            with self.log_output:
                print("💥 Error al descargar el modelo.")
            return False

        vae_file = MODELS_DIR / "sdxl_vae.safetensors"
        if not vae_file.exists():
            with self.log_output:
                print(f"🌐 Descargando VAE en {vae_file} ...")
            vae_cmd = [
                "aria2c",
                "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                "--console-log-level=warn",
                "-c",
                "-s",
                "16",
                "-x",
                "16",
                "-k",
                "10M",
                "-d",
                str(MODELS_DIR),
                "-o",
                vae_file.name,
            ]
            if self._run_command(vae_cmd) != 0:
                with self.log_output:
                    print("💥 Error al descargar el VAE.")
                return False

        if self.model_file.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors  # type: ignore

                load_safetensors(self.model_file)
            except Exception:
                new_model_file = self.model_file.with_suffix(".ckpt")
                self.model_file.rename(new_model_file)
                self.model_file = new_model_file
                with self.log_output:
                    print(f"Renombrado modelo a {self.model_file}")

        if self.model_file.suffix.lower() == ".ckpt":
            try:
                import torch  # type: ignore
            except Exception as exc:
                with self.log_output:
                    print(f"⚠️ No se pudo validar el checkpoint (torch no disponible: {exc}).")
            else:
                try:
                    torch.load(self.model_file, map_location="cpu")
                except Exception:
                    with self.log_output:
                        print("💥 Error al validar el checkpoint descargado.")
                    return False

        self.previous_model_url = url
        return True

    # ------------------------------------------------------------------
    def create_dataset_config(self) -> Path:
        cfg = self.config
        config_dir = cfg.output_dir / "config"
        self._ensure_dir(config_dir)
        dataset_config_file = config_dir / "dataset_config.toml"
        if cfg.custom_dataset:
            dataset_config_file.write_text(cfg.custom_dataset, encoding="utf-8")
            return dataset_config_file

        subsets = [
            {
                "image_dir": str(cfg.dataset_dir),
                "caption_extension": cfg.caption_extension,
                "shuffle_caption": cfg.shuffle_tags,
                "keep_tokens": int(cfg.activation_tags),
                "num_repeats": int(cfg.repeats),
            }
        ]
        data = {
            "datasets": [
                {
                    "resolution": cfg.resolution,
                    "enable_bucket": True,
                    "flip_aug": cfg.flip_aug,
                    "color_aug": False,
                    "random_crop": False,
                    "cache_latents": cfg.cache_latents,
                    "keep_tokens": int(cfg.activation_tags),
                    "caption_extension": cfg.caption_extension,
                    "shuffle_caption": cfg.shuffle_tags,
                    "subsets": subsets,
                }
            ]
        }
        dataset_config_file.write_text(toml.dumps(data), encoding="utf-8")
        return dataset_config_file

    def create_training_config(self, dataset_config_path: Path, resolved_scheduler: Optional[str] = None) -> Path:
        cfg = self.config
        config_dir = cfg.output_dir / "config"
        self._ensure_dir(config_dir)
        training_config_file = config_dir / "training_config.toml"

        optimizer_args = cfg.optimizer_args[:] if cfg.optimizer_args else []
        lr_scheduler_args = cfg.lr_scheduler_args[:] if cfg.lr_scheduler_args else None
        if cfg.recommended_values:
            if any(opt in cfg.optimizer.lower() for opt in ["dadapt", "prodigy"]):
                cfg.unet_lr = 0.75
                cfg.text_encoder_lr = 0.75
                cfg.network_alpha = cfg.network_dim
                cfg.full_precision = False
            if cfg.optimizer == "Prodigy":
                optimizer_args = [
                    "decouple=True",
                    "weight_decay=0.01",
                    "betas=[0.9,0.999]",
                    "d_coef=2",
                    "use_bias_correction=True",
                    "safeguard_warmup=True",
                ]
            elif cfg.optimizer == "AdamW8bit":
                optimizer_args = ["weight_decay=0.1", "betas=[0.9,0.99]"]
            elif cfg.optimizer == "AdaFactor":
                optimizer_args = [
                    "scale_parameter=False",
                    "relative_step=False",
                    "warmup_init=False",
                ]
            elif cfg.optimizer == "Came":
                optimizer_args = ["weight_decay=0.04"]

        if cfg.lora_type.lower() == "locon":
            network_args = [f"conv_dim={cfg.conv_dim}", f"conv_alpha={cfg.conv_alpha}"]
        else:
            network_args = None

        lr_scheduler_num_cycles = cfg.lr_scheduler_number
        lr_scheduler_power = cfg.lr_scheduler_number

        lr_scheduler_type = cfg.lr_scheduler_type
        lr_scheduler = resolved_scheduler or cfg.lr_scheduler

        training_dict: Dict[str, Any] = {
            "network_arguments": {
                "unet_lr": cfg.unet_lr,
                "text_encoder_lr": cfg.text_encoder_lr if not cfg.cache_text_encoder_outputs else 0,
                "network_dim": cfg.network_dim,
                "network_alpha": cfg.network_alpha,
                "network_module": "networks.lora",
                "network_args": network_args,
                "network_train_unet_only": cfg.text_encoder_lr == 0 or cfg.cache_text_encoder_outputs,
            },
            "optimizer_arguments": {
                "learning_rate": cfg.unet_lr,
                "lr_scheduler": lr_scheduler,
                "lr_scheduler_type": lr_scheduler_type,
                "lr_scheduler_args": lr_scheduler_args,
                "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
                "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
                "lr_warmup_steps": cfg.lr_warmup_steps if lr_scheduler not in ("cosine", "constant") else None,
                "optimizer_type": cfg.optimizer,
                "optimizer_args": optimizer_args or None,
                "loss_type": "l2",
                "max_grad_norm": 1.0,
            },
            "training_arguments": {
                "lowram": cfg.low_ram,
                "pretrained_model_name_or_path": str(self.model_file) if self.model_file else cfg.model_path,
                "vae": str(MODELS_DIR / "sdxl_vae.safetensors"),
                "max_train_steps": None,
                "max_train_epochs": cfg.epochs,
                "train_batch_size": cfg.batch_size,
                "seed": cfg.seed,
                "max_token_length": cfg.max_token_length,
                "xformers": cfg.cross_attention == "xformers",
                "sdpa": cfg.cross_attention == "sdpa",
                "min_snr_gamma": cfg.min_snr_gamma if cfg.min_snr_gamma_enabled else None,
                "ip_noise_gamma": cfg.ip_noise_gamma if cfg.ip_noise_gamma_enabled else None,
                "no_half_vae": True,
                "gradient_checkpointing": True,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "max_data_loader_n_workers": 1,
                "persistent_data_loader_workers": True,
                "mixed_precision": "fp16" if "fp16" in cfg.precision else "bf16" if "bf16" in cfg.precision else "no",
                "full_fp16": "full" in cfg.precision and "fp16" in cfg.precision,
                "full_bf16": "full" in cfg.precision and "bf16" in cfg.precision,
                "cache_latents": cfg.cache_latents,
                "cache_latents_to_disk": cfg.cache_latents_to_disk,
                "cache_text_encoder_outputs": cfg.cache_text_encoder_outputs,
                "min_timestep": 0,
                "max_timestep": 1000,
                "prior_loss_weight": 1.0,
                "multires_noise_iterations": 6 if cfg.multinoise else None,
                "multires_noise_discount": 0.3 if cfg.multinoise else None,
                "v_parameterization": cfg.custom_model_is_vpred or None,
                "clip_skip": None,
                "use_wandb": bool(cfg.wandb_key),
                "log_prefix": cfg.project_name,
                "log_folder": str(cfg.output_dir / "_logs"),
                "save_every_n_epochs": cfg.save_every_n_epochs,
                "keep_only_last_n_epochs": cfg.keep_only_last_n_epochs,
                "output_dir": str(cfg.output_dir),
            },
            "saving_arguments": {
                "output_dir": str(cfg.output_dir),
                "save_model_as": "safetensors",
                "network_weights_only": True,
            },
            "additional_parameters": {
                "dataset_config": str(dataset_config_path),
            },
        }
        training_config_file.write_text(toml.dumps(training_dict), encoding="utf-8")
        return training_config_file

    # ------------------------------------------------------------------
    def calculate_rex_steps(self, dataset_config: Path) -> None:
        cfg = self.config
        subsets = toml.loads(dataset_config.read_text(encoding="utf-8"))["datasets"][0]["subsets"]
        supported_types = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        from library.train_util import BucketManager  # type: ignore

        bucket_manager = BucketManager(
            False,
            (cfg.resolution, cfg.resolution),
            cfg.min_bucket_reso,
            cfg.max_bucket_reso,
            cfg.bucket_reso_steps,
        )
        bucket_manager.make_buckets()
        with self.log_output:
            for subset in tqdm(subsets, desc="REx buckets", leave=False):
                image_dir = Path(subset["image_dir"])
                for image in image_dir.iterdir():
                    if image.suffix.lower() not in supported_types:
                        continue
                    with Image.open(image) as img:
                        bucket_reso, _, _ = bucket_manager.select_bucket(img.width, img.height)
                    for _ in trange(
                        subset["num_repeats"], desc=image.name, leave=False, colour="#4caf50"
                    ):
                        bucket_manager.add_image(bucket_reso, image)
        steps_before_acc = sum(math.ceil(len(bucket) / cfg.batch_size) for bucket in bucket_manager.buckets)
        calculated_max_steps = math.ceil(steps_before_acc / cfg.gradient_accumulation_steps) * cfg.epochs
        if self.config.lr_scheduler_args is None:
            self.config.lr_scheduler_args = []
        cycle_steps = calculated_max_steps // (cfg.lr_scheduler_num_cycles or 1)
        self.config.lr_scheduler_args.append(f"first_cycle_max_steps={cycle_steps}")
        warmup_steps = round(calculated_max_steps * cfg.lr_warmup_ratio) // (cfg.lr_scheduler_num_cycles or 1)
        if warmup_steps > 0:
            self.config.lr_scheduler_args.append(f"warmup_steps={warmup_steps}")

    # ------------------------------------------------------------------
    def run_training(self) -> None:
        self.configure_logging()
        cfg = self.config
        cfg.derive()

        for directory in (
            MODELS_DIR,
            DOWNLOADS_DIR,
            cfg.output_dir,
        ):
            self._ensure_dir(directory)

        if not self.validate_dataset():
            return

        if not self.dependencies_installed:
            start = time.time()
            self.install_trainer()
            elapsed = int(time.time() - start)
            with self.log_output:
                print(f"✅ Instalación completada en {elapsed} s.")
        else:
            with self.log_output:
                print("✅ Dependencias ya instaladas.")

        if not self.download_model():
            return

        resolved_scheduler = self.prepare_scheduler()
        dataset_config = self.create_dataset_config()
        if cfg.lr_scheduler_type:
            self.calculate_rex_steps(dataset_config)
        training_config = self.create_training_config(dataset_config, resolved_scheduler)

        with self.log_output:
            print("⭐ Iniciando entrenamiento...")

        cmd = [
            str(DEFAULT_VENV_PYTHON),
            str(TRAIN_NETWORK),
            f"--config_file={training_config}",
            f"--dataset_config={dataset_config}",
        ]
        result = self._run_command(cmd, cwd=KOHYA_DIR)
        if result == 0:
            with self.log_output:
                display(Markdown(f"### ✅ ¡Hecho! Tus archivos se encuentran en `{cfg.output_dir}`"))
        else:
            with self.log_output:
                print(f"💥 Entrenamiento finalizó con código {result}")

    # ------------------------------------------------------------------
    def build_ui(self) -> widgets.Widget:
        cfg = self.config

        dataset_dir = widgets.Text(
            value=str(cfg.dataset_dir), description="dataset_dir", layout=widgets.Layout(width="100%")
        )
        output_dir = widgets.Text(
            value=str(cfg.output_dir), description="output_dir", layout=widgets.Layout(width="100%")
        )
        model_path = widgets.Text(
            value=str(cfg.model_path), description="model_path", layout=widgets.Layout(width="100%")
        )
        epochs = widgets.IntSlider(value=cfg.epochs, min=1, max=200, description="epochs")
        repeats = widgets.IntText(value=cfg.repeats, description="repeats")
        batch_size = widgets.IntText(value=cfg.batch_size, description="batch_size")
        unet_lr = widgets.FloatText(value=cfg.unet_lr, description="unet_lr")
        text_encoder_lr = widgets.FloatText(value=cfg.text_encoder_lr, description="text_enc_lr")
        num_images = widgets.IntText(value=cfg.num_images, description="num_images")
        steps_display = widgets.HTML()

        def update_steps(*_args: Any) -> None:
            value = 0
            if batch_size.value:
                value = (num_images.value * repeats.value * epochs.value) // batch_size.value
            steps_display.value = f"<b>steps</b>: {value}"

        for control in (epochs, repeats, batch_size, num_images):
            control.observe(update_steps, names="value")
        update_steps()

        advanced_controls = self._build_advanced_controls()
        run_button = widgets.Button(description="Run", button_style="success", icon="play")

        def on_run_click(_btn: widgets.Button) -> None:
            self.log_output.clear_output()
            cfg.dataset_dir = Path(dataset_dir.value.strip())
            cfg.output_dir = Path(output_dir.value.strip())
            cfg.model_path = model_path.value.strip()
            cfg.epochs = int(epochs.value)
            cfg.repeats = int(repeats.value)
            cfg.batch_size = int(batch_size.value)
            cfg.unet_lr = float(unet_lr.value)
            cfg.text_encoder_lr = float(text_encoder_lr.value)
            cfg.num_images = int(num_images.value)
            cfg.preferred_unit = "Epochs"
            cfg.how_many = cfg.epochs
            with self.log_output:
                print(json.dumps({
                    "dataset_dir": str(cfg.dataset_dir),
                    "output_dir": str(cfg.output_dir),
                    "model_path": cfg.model_path,
                    "epochs": cfg.epochs,
                    "repeats": cfg.repeats,
                    "batch_size": cfg.batch_size,
                    "unet_lr": cfg.unet_lr,
                    "text_encoder_lr": cfg.text_encoder_lr,
                }, indent=2))
            self.run_training()

        run_button.on_click(on_run_click)

        accordion = widgets.Accordion(children=[advanced_controls])
        accordion.set_title(0, "Avanzado")
        accordion.selected_index = None

        layout = widgets.VBox(
            [
                dataset_dir,
                output_dir,
                model_path,
                widgets.HBox([epochs, repeats, batch_size]),
                widgets.HBox([unet_lr, text_encoder_lr, num_images]),
                steps_display,
                run_button,
                accordion,
                self.log_output,
            ]
        )
        return layout

    def _build_advanced_controls(self) -> widgets.Widget:
        cfg = self.config
        project_name = widgets.Text(value=cfg.project_name, description="project")
        resolution = widgets.IntText(value=cfg.resolution, description="resolution")
        caption_extension = widgets.Dropdown(
            options=[".txt", ".caption"], value=cfg.caption_extension, description="caption_ext"
        )
        shuffle_tags = widgets.Checkbox(value=cfg.shuffle_tags, description="shuffle_tags")
        activation_tags = widgets.IntSlider(value=cfg.activation_tags, min=0, max=3, description="keep_tokens")
        flip_aug = widgets.Checkbox(value=cfg.flip_aug, description="flip_aug")
        optimizer = widgets.Dropdown(
            options=[
                "AdamW8bit",
                "Prodigy",
                "DAdaptation",
                "DadaptAdam",
                "DadaptLion",
                "AdamW",
                "AdaFactor",
                "Came",
            ],
            value=cfg.optimizer,
            description="optimizer",
        )
        lora_type = widgets.Dropdown(options=["LoRA", "LoCon"], value=cfg.lora_type, description="lora_type")
        network_dim = widgets.IntText(value=cfg.network_dim, description="network_dim")
        network_alpha = widgets.IntText(value=cfg.network_alpha, description="network_alpha")
        lr_scheduler = widgets.Dropdown(
            options=[
                "constant",
                "cosine",
                "cosine_with_restarts",
                "constant_with_warmup",
                "rex",
            ],
            value=cfg.lr_scheduler,
            description="scheduler",
        )
        lr_scheduler_number = widgets.IntText(value=cfg.lr_scheduler_number, description="sched_num")
        lr_warmup_ratio = widgets.FloatSlider(
            value=cfg.lr_warmup_ratio, min=0.0, max=0.2, step=0.01, description="warmup_ratio"
        )
        save_every = widgets.IntText(value=cfg.save_every_n_epochs, description="save_every")
        keep_last = widgets.IntText(value=cfg.keep_only_last_n_epochs, description="keep_last")

        def apply(_=None) -> None:
            cfg.project_name = project_name.value.strip() or cfg.project_name
            cfg.resolution = int(resolution.value)
            cfg.caption_extension = caption_extension.value
            cfg.shuffle_tags = bool(shuffle_tags.value)
            cfg.activation_tags = int(activation_tags.value)
            cfg.flip_aug = bool(flip_aug.value)
            cfg.optimizer = optimizer.value
            cfg.lora_type = lora_type.value
            cfg.network_dim = int(network_dim.value)
            cfg.network_alpha = int(network_alpha.value)
            cfg.lr_scheduler = lr_scheduler.value
            cfg.lr_scheduler_number = int(lr_scheduler_number.value)
            cfg.lr_warmup_ratio = float(lr_warmup_ratio.value)
            cfg.save_every_n_epochs = int(save_every.value)
            cfg.keep_only_last_n_epochs = int(keep_last.value)

        apply_button = widgets.Button(description="Aplicar", button_style="info", icon="check")
        apply_button.on_click(apply)

        return widgets.VBox(
            [
                project_name,
                resolution,
                caption_extension,
                shuffle_tags,
                activation_tags,
                flip_aug,
                optimizer,
                lora_type,
                network_dim,
                network_alpha,
                lr_scheduler,
                lr_scheduler_number,
                lr_warmup_ratio,
                save_every,
                keep_last,
                apply_button,
            ]
        )


def launch() -> widgets.Widget:
    """Return the interactive UI widget."""
    app = TrainerApp()
    return app.build_ui()
