from __future__ import annotations

import importlib
import importlib.util
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

from .compat import ensure_torchvision_functional_tensor
from .config import Settings


@dataclass
class TranslatorBundle:
    tokenizer: object
    model: object


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, settings: Settings) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings: Settings):
        if getattr(self, "_initialized", False):
            return
        self.settings = settings
        if self.settings.local_models_only:
            # Force local-cache resolution to avoid long network stalls in offline/restricted environments.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        except Exception:
            self.torch = None
            self.device = "cpu"
            self.dtype = None
        self._manga_ocr = None
        self._easyocr = None
        self._translator_ja_en: Optional[TranslatorBundle] = None
        self._translator_ko_en: Optional[TranslatorBundle] = None
        self._m2m = None
        self._m2m_tok = None
        self._inpaint_pipe = None
        self._img2img_pipe = None
        self._quality_img2img_pipe = None
        self._controlnet_img2img = None
        self._controlnet_modes: list[str] = []
        self._quality_controlnet_img2img = None
        self._quality_controlnet_modes: list[str] = []
        self._llm = None
        self._capability_cache: Dict[str, bool] = {}
        self._inpaint_load_failed = False
        self._img2img_load_failed = False
        self._quality_img2img_load_failed = False
        self._controlnet_load_failed = False
        self._quality_controlnet_load_failed = False
        self._initialized = True
        logger.info("ModelManager initialized on device={}", self.device)

    def _resolve_model_locator(self, configured: str | Path, local_subdir: str) -> Optional[str]:
        configured_path = Path(str(configured))
        if configured_path.exists():
            return str(configured_path)
        if self.settings.local_models_only:
            local_path = self.settings.models_dir / local_subdir
            if local_path.exists():
                return str(local_path)
            return None
        return str(configured)

    def _has_model_files(self, configured: str | Path, local_subdir: str) -> bool:
        configured_path = Path(str(configured))
        if configured_path.exists():
            return True
        if not self.settings.local_models_only:
            return True
        return (self.settings.models_dir / local_subdir).exists()

    def module_available(self, module_name: str) -> bool:
        if module_name in self._capability_cache:
            return self._capability_cache[module_name]
        spec = importlib.util.find_spec(module_name)
        available = spec is not None
        # Some modules are importable by spec but fail at runtime because of
        # transitive dependency mismatches (for example, realesrgan -> basicsr -> torchvision).
        # Probe-import these modules once to prevent false capability positives.
        if available and module_name in {"realesrgan", "facenet_pytorch"}:
            if module_name == "realesrgan":
                ensure_torchvision_functional_tensor()
            try:
                importlib.import_module(module_name)
            except Exception:
                available = False
        self._capability_cache[module_name] = available
        return available

    def capabilities(self, refresh: bool = False) -> dict:
        if refresh:
            self._capability_cache.clear()
        modules = {
            "torch": self.module_available("torch"),
            "diffusers": self.module_available("diffusers"),
            "manga_ocr": self.module_available("manga_ocr"),
            "easyocr": self.module_available("easyocr"),
            "transformers": self.module_available("transformers"),
            "facenet_pytorch": self.module_available("facenet_pytorch"),
            "controlnet_aux": self.module_available("controlnet_aux"),
            "realesrgan": self.module_available("realesrgan"),
            "lama_cleaner": self.module_available("lama_cleaner"),
        }
        model_files = {
            "manga_ocr": self._has_model_files("kha-white/manga-ocr-base", "manga_ocr"),
            "easyocr": self._has_model_files("easyocr", "easyocr"),
            "marian_ja_en": self._has_model_files(self.settings.marian_ja_en, "marian_ja_en"),
            "marian_ko_en": self._has_model_files(self.settings.marian_ko_en, "marian_ko_en"),
            "m2m100": self._has_model_files(self.settings.m2m100_fallback, "m2m100"),
            "sd_inpaint": self._has_model_files(self.settings.sd_inpaint_model, "sd_inpaint"),
            "sd_img2img": self._has_model_files(self.settings.sd_img2img_model, "sd_base"),
            "quality_sdxl": self._has_model_files(self.settings.quality_sdxl_model, "quality_sdxl"),
            "controlnet_canny": self._has_model_files(self.settings.controlnet_canny_model, "controlnet_canny"),
            "controlnet_depth": self._has_model_files(self.settings.controlnet_depth_model, "controlnet_depth"),
            "controlnet_openpose": self._has_model_files(
                self.settings.controlnet_openpose_model,
                "controlnet_openpose",
            ),
            "quality_controlnet_canny": self._has_model_files(
                self.settings.quality_controlnet_canny_model, "quality_controlnet_canny"
            ),
            "quality_controlnet_depth": self._has_model_files(
                self.settings.quality_controlnet_depth_model, "quality_controlnet_depth"
            ),
            "quality_controlnet_openpose": self._has_model_files(
                self.settings.quality_controlnet_openpose_model, "quality_controlnet_openpose"
            ),
        }
        loaded = {
            "ocr_loaded": self._manga_ocr is not None or self._easyocr is not None,
            "translator_loaded": self._translator_ja_en is not None or self._translator_ko_en is not None,
            "inpaint_loaded": self._inpaint_pipe is not None,
            "img2img_loaded": self._img2img_pipe is not None,
            "llm_loaded": self._llm is not None,
        }
        return {
            "device": self.device,
            "gpu": self.gpu_info(),
            "modules": modules,
            "model_files": model_files,
            "loaded": loaded,
            "local_models_only": self.settings.local_models_only,
        }

    def _optimize_diffusion_pipe(self, pipe: Any) -> Any:
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        xformers_enabled = False
        if self.device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                xformers_enabled = True
            except Exception:
                xformers_enabled = False

        if not xformers_enabled:
            try:
                pipe.enable_attention_slicing("max")
            except Exception:
                pass

        if self.device == "cuda":
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
            try:
                if self.torch is not None and hasattr(pipe, "unet") and pipe.unet is not None:
                    pipe.unet.to(memory_format=self.torch.channels_last)
            except Exception:
                pass

        try:
            from diffusers import DPMSolverMultistepScheduler

            if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
        except Exception:
            pass

        return pipe

    def missing_requirements(
        self,
        render_quality: str = "balanced",
        need_translation: bool = True,
        need_ocr: bool = True,
        need_diffusion: bool = False,
    ) -> list[str]:
        caps = self.capabilities()
        missing = []
        modules = caps["modules"]
        model_files = caps["model_files"]
        quality_mode = render_quality == "quality"

        if quality_mode and self.settings.quality_require_gpu and self.device != "cuda":
            missing.append("quality_requires_cuda_gpu")

        if self.settings.strict_require_gpu and self.device != "cuda":
            missing.append("cuda_gpu")
        if self.settings.strict_require_diffusion or need_diffusion or render_quality in {"final", "quality"}:
            if not modules.get("diffusers", False):
                missing.append("diffusers")
            elif self.settings.local_models_only and not model_files.get("sd_img2img", False):
                missing.append("sd_img2img_model_files")
            if quality_mode and self.settings.quality_require_full_path and self.settings.local_models_only:
                if not model_files.get("quality_sdxl", False):
                    missing.append("quality_sdxl_model_files")
                if not model_files.get("quality_controlnet_canny", False):
                    missing.append("quality_controlnet_canny_model_files")
                if not model_files.get("quality_controlnet_depth", False):
                    missing.append("quality_controlnet_depth_model_files")
                if not model_files.get("quality_controlnet_openpose", False):
                    missing.append("quality_controlnet_openpose_model_files")
            if (
                self.settings.strict_pro_mode
                and self.settings.strict_require_diffusion
                and render_quality in {"final", "quality"}
                and self.device != "cuda"
            ):
                missing.append("cuda_gpu_for_final_diffusion")
        if self.settings.strict_require_ocr or need_ocr:
            if not modules.get("manga_ocr", False) and not modules.get("easyocr", False):
                missing.append("manga_ocr_or_easyocr")
            elif self.settings.local_models_only and not (
                model_files.get("manga_ocr", False) or model_files.get("easyocr", False)
            ):
                missing.append("ocr_model_files")
        if self.settings.strict_require_translation_models or need_translation:
            if not modules.get("transformers", False):
                missing.append("transformers")
            elif self.settings.local_models_only and not (
                model_files.get("marian_ja_en", False)
                and model_files.get("marian_ko_en", False)
                or model_files.get("m2m100", False)
            ):
                missing.append("translation_model_files")
        if self.settings.enforce_identity_consistency and render_quality in {"balanced", "final", "quality"}:
            if not modules.get("facenet_pytorch", False):
                missing.append("facenet_pytorch")
        if self.settings.enable_diffusion_inpaint and (self.settings.strict_require_diffusion or need_diffusion):
            if not modules.get("lama_cleaner", False) and not modules.get("diffusers", False):
                missing.append("lama_cleaner_or_diffusers")
            elif self.settings.local_models_only and not (
                model_files.get("sd_inpaint", False) or model_files.get("sd_img2img", False)
            ):
                missing.append("sd_inpaint_model_files")
        return sorted(set(missing))

    def strict_mode_blockers(
        self,
        render_quality: str = "balanced",
        need_translation: bool = True,
        need_ocr: bool = True,
        need_diffusion: bool = False,
    ) -> list[str]:
        if not self.settings.strict_pro_mode:
            return []
        return self.missing_requirements(
            render_quality=render_quality,
            need_translation=need_translation,
            need_ocr=need_ocr,
            need_diffusion=need_diffusion,
        )

    def load_manga_ocr(self):
        if self._manga_ocr is not None:
            return self._manga_ocr
        try:
            from manga_ocr import MangaOcr

            model_locator = self._resolve_model_locator(
                configured="kha-white/manga-ocr-base",
                local_subdir="manga_ocr",
            )
            if model_locator is None:
                logger.warning("Skipping manga-ocr load (local_models_only and model files missing)")
                return None
            self._manga_ocr = MangaOcr(
                pretrained_model_name_or_path=model_locator,
                force_cpu=self.device != "cuda",
            )
            logger.info("Loaded manga-ocr model")
        except Exception as exc:
            logger.warning("manga-ocr unavailable: {}", exc)
            self._manga_ocr = None
        return self._manga_ocr

    def load_easyocr(self):
        if self._easyocr is not None:
            return self._easyocr
        try:
            import easyocr

            easyocr_dir = self.settings.models_dir / "easyocr"
            if self.settings.local_models_only and not easyocr_dir.exists():
                logger.warning("Skipping EasyOCR load (local_models_only and model files missing)")
                return None
            easyocr_dir.mkdir(parents=True, exist_ok=True)
            self._easyocr = easyocr.Reader(
                ["ko", "en", "ja"],
                gpu=self.device == "cuda",
                download_enabled=not self.settings.local_models_only,
                model_storage_directory=str(easyocr_dir),
                user_network_directory=str(easyocr_dir),
            )
            logger.info("Loaded EasyOCR reader")
        except Exception as exc:
            logger.warning("EasyOCR unavailable: {}", exc)
            self._easyocr = None
        return self._easyocr

    def _load_translator(self, model_id: str, local_subdir: str) -> Optional[TranslatorBundle]:
        locator = self._resolve_model_locator(model_id, local_subdir=local_subdir)
        if locator is None:
            logger.warning("Skipping translator load for {} (local model files missing)", model_id)
            return None
        try:
            from transformers import MarianMTModel, MarianTokenizer

            tokenizer = MarianTokenizer.from_pretrained(
                locator,
                local_files_only=self.settings.local_models_only,
            )
            model = MarianMTModel.from_pretrained(
                locator,
                local_files_only=self.settings.local_models_only,
            ).to(self.device)
            return TranslatorBundle(tokenizer=tokenizer, model=model)
        except Exception as exc:
            logger.warning("Failed to load translator {}: {}", model_id, exc)
            return None

    def load_ja_en(self) -> Optional[TranslatorBundle]:
        if self._translator_ja_en is None:
            self._translator_ja_en = self._load_translator(
                self.settings.marian_ja_en,
                local_subdir="marian_ja_en",
            )
            if self._translator_ja_en:
                logger.info("Loaded JA->EN translator")
        return self._translator_ja_en

    def load_ko_en(self) -> Optional[TranslatorBundle]:
        if self._translator_ko_en is None:
            self._translator_ko_en = self._load_translator(
                self.settings.marian_ko_en,
                local_subdir="marian_ko_en",
            )
            if self._translator_ko_en:
                logger.info("Loaded KO->EN translator")
        return self._translator_ko_en

    def load_m2m100(self):
        if self._m2m is not None and self._m2m_tok is not None:
            return self._m2m_tok, self._m2m
        locator = self._resolve_model_locator(self.settings.m2m100_fallback, local_subdir="m2m100")
        if locator is None:
            logger.warning("Skipping M2M100 load (local model files missing)")
            return None, None
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            self._m2m_tok = M2M100Tokenizer.from_pretrained(
                locator,
                local_files_only=self.settings.local_models_only,
            )
            self._m2m = M2M100ForConditionalGeneration.from_pretrained(
                locator,
                local_files_only=self.settings.local_models_only,
            ).to(self.device)
            logger.info("Loaded M2M100 fallback translator")
            return self._m2m_tok, self._m2m
        except Exception as exc:
            logger.warning("M2M100 fallback unavailable: {}", exc)
            return None, None

    def load_llm(self):
        if self._llm is not None:
            return self._llm
        if not self.settings.enable_llm_post_edit:
            return None
        try:
            from llama_cpp import Llama

            if Path(self.settings.llm_gguf_path).exists():
                self._llm = Llama(
                    model_path=str(self.settings.llm_gguf_path),
                    n_ctx=4096,
                    n_gpu_layers=-1 if self.device == "cuda" else 0,
                )
                logger.info("Loaded local LLM post-editor")
        except Exception as exc:
            logger.warning("LLM post-editor unavailable: {}", exc)
            self._llm = None
        return self._llm

    def load_inpaint_pipeline(self):
        if self._inpaint_pipe is not None:
            return self._inpaint_pipe
        if self._inpaint_load_failed:
            return None
        if not self.settings.enable_diffusion_inpaint:
            return None
        locator = self._resolve_model_locator(self.settings.sd_inpaint_model, local_subdir="sd_inpaint")
        if locator is None:
            logger.warning("Skipping inpaint diffusion load (local model files missing)")
            self._inpaint_load_failed = True
            return None
        try:
            from diffusers import AutoPipelineForInpainting

            pipe = AutoPipelineForInpainting.from_pretrained(
                locator,
                torch_dtype=self.dtype if self.device == "cuda" else None,
                local_files_only=self.settings.local_models_only,
            )
            if self.device == "cuda":
                pipe = pipe.to(self.device)
            pipe = self._optimize_diffusion_pipe(pipe)
            self._inpaint_pipe = pipe
            self._inpaint_load_failed = False
            logger.info("Loaded diffusion inpainting pipeline")
        except Exception as exc:
            logger.warning("Diffusion inpaint pipeline unavailable: {}", exc)
            self._inpaint_pipe = None
            self._inpaint_load_failed = True
        return self._inpaint_pipe

    def load_img2img_pipeline(self, quality_mode: bool = False):
        if quality_mode:
            if self._quality_img2img_pipe is not None:
                return self._quality_img2img_pipe
            if self._quality_img2img_load_failed:
                return None
            if self.settings.quality_require_gpu and self.device != "cuda":
                return None
            model_id = self.settings.quality_sdxl_model
            subdir = "quality_sdxl"
        else:
            if self._img2img_pipe is not None:
                return self._img2img_pipe
            if self._img2img_load_failed:
                return None
            model_id = self.settings.sd_img2img_model
            subdir = "sd_base"

        if not self.settings.enable_diffusion_styles:
            return None

        locator = self._resolve_model_locator(model_id, local_subdir=subdir)
        if locator is None:
            logger.warning("Skipping image2image diffusion load (local model files missing)")
            if quality_mode:
                self._quality_img2img_load_failed = True
            else:
                self._img2img_load_failed = True
            return None
        try:
            from diffusers import AutoPipelineForImage2Image

            pipe = AutoPipelineForImage2Image.from_pretrained(
                locator,
                torch_dtype=self.dtype if self.device == "cuda" else None,
                safety_checker=None,
                local_files_only=self.settings.local_models_only,
            )
            if self.device == "cuda":
                pipe = pipe.to(self.device)
            pipe = self._optimize_diffusion_pipe(pipe)
            if quality_mode:
                self._quality_img2img_pipe = pipe
                self._quality_img2img_load_failed = False
                logger.info("Loaded quality diffusion image2image pipeline (SDXL path)")
                return self._quality_img2img_pipe
            self._img2img_pipe = pipe
            self._img2img_load_failed = False
            logger.info("Loaded diffusion image2image pipeline")
            return self._img2img_pipe
        except Exception as exc:
            logger.warning("Diffusion image2image unavailable: {}", exc)
            if quality_mode:
                self._quality_img2img_pipe = None
                self._quality_img2img_load_failed = True
                return None
            self._img2img_pipe = None
            self._img2img_load_failed = True
            return None

    def load_controlnet_img2img_pipeline(self, quality_mode: bool = False) -> Optional[dict[str, Any]]:
        if quality_mode:
            if self._quality_controlnet_img2img is not None:
                return {"pipe": self._quality_controlnet_img2img, "modes": list(self._quality_controlnet_modes)}
            if self._quality_controlnet_load_failed:
                return None
            if self.settings.quality_require_gpu and self.device != "cuda":
                return None
            if not self.settings.enable_diffusion_styles:
                return None
            base_locator = self._resolve_model_locator(self.settings.quality_sdxl_model, local_subdir="quality_sdxl")
            if base_locator is None:
                self._quality_controlnet_load_failed = True
                return None
            canny_locator = self._resolve_model_locator(
                self.settings.quality_controlnet_canny_model,
                local_subdir="quality_controlnet_canny",
            )
            depth_locator = self._resolve_model_locator(
                self.settings.quality_controlnet_depth_model,
                local_subdir="quality_controlnet_depth",
            )
            pose_locator = self._resolve_model_locator(
                self.settings.quality_controlnet_openpose_model,
                local_subdir="quality_controlnet_openpose",
            )
            pipeline_name = "StableDiffusionXLControlNetImg2ImgPipeline"
        else:
            if self._controlnet_img2img is not None:
                return {"pipe": self._controlnet_img2img, "modes": list(self._controlnet_modes)}
            if self._controlnet_load_failed:
                return None
            if not self.settings.enable_diffusion_styles:
                return None
            base_locator = self._resolve_model_locator(self.settings.sd_img2img_model, local_subdir="sd_base")
            if base_locator is None:
                self._controlnet_load_failed = True
                return None
            canny_locator = self._resolve_model_locator(
                self.settings.controlnet_canny_model,
                local_subdir="controlnet_canny",
            )
            depth_locator = self._resolve_model_locator(
                self.settings.controlnet_depth_model,
                local_subdir="controlnet_depth",
            )
            pose_locator = self._resolve_model_locator(
                self.settings.controlnet_openpose_model,
                local_subdir="controlnet_openpose",
            )
            pipeline_name = "StableDiffusionControlNetImg2ImgPipeline"

        controls = []
        modes: list[str] = []
        try:
            from diffusers import ControlNetModel
            from diffusers import StableDiffusionControlNetImg2ImgPipeline
            from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

            pipeline_cls = (
                StableDiffusionXLControlNetImg2ImgPipeline
                if pipeline_name == "StableDiffusionXLControlNetImg2ImgPipeline"
                else StableDiffusionControlNetImg2ImgPipeline
            )

            if canny_locator is not None:
                controls.append(
                    ControlNetModel.from_pretrained(
                        canny_locator,
                        torch_dtype=self.dtype if self.device == "cuda" else None,
                        local_files_only=self.settings.local_models_only,
                    )
                )
                modes.append("canny")
            if depth_locator is not None:
                controls.append(
                    ControlNetModel.from_pretrained(
                        depth_locator,
                        torch_dtype=self.dtype if self.device == "cuda" else None,
                        local_files_only=self.settings.local_models_only,
                    )
                )
                modes.append("depth")
            if pose_locator is not None:
                controls.append(
                    ControlNetModel.from_pretrained(
                        pose_locator,
                        torch_dtype=self.dtype if self.device == "cuda" else None,
                        local_files_only=self.settings.local_models_only,
                    )
                )
                modes.append("pose")

            if not controls:
                if quality_mode:
                    self._quality_controlnet_load_failed = True
                else:
                    self._controlnet_load_failed = True
                return None

            controlnet = controls[0] if len(controls) == 1 else controls
            pipe = pipeline_cls.from_pretrained(
                base_locator,
                controlnet=controlnet,
                torch_dtype=self.dtype if self.device == "cuda" else None,
                safety_checker=None,
                local_files_only=self.settings.local_models_only,
            )
            if self.device == "cuda":
                pipe = pipe.to(self.device)
            pipe = self._optimize_diffusion_pipe(pipe)

            if quality_mode:
                self._quality_controlnet_img2img = pipe
                self._quality_controlnet_modes = modes
                self._quality_controlnet_load_failed = False
                logger.info("Loaded quality controlnet image2image pipeline with modes={}", modes)
                return {"pipe": self._quality_controlnet_img2img, "modes": list(self._quality_controlnet_modes)}

            self._controlnet_img2img = pipe
            self._controlnet_modes = modes
            self._controlnet_load_failed = False
            logger.info("Loaded controlnet image2image pipeline with modes={}", modes)
            return {"pipe": self._controlnet_img2img, "modes": list(self._controlnet_modes)}
        except Exception as exc:
            logger.warning("ControlNet image2image unavailable: {}", exc)
            if quality_mode:
                self._quality_controlnet_img2img = None
                self._quality_controlnet_modes = []
                self._quality_controlnet_load_failed = True
                return None
            self._controlnet_img2img = None
            self._controlnet_modes = []
            self._controlnet_load_failed = True
            return None

    def translate_with_bundle(self, bundle: TranslatorBundle, texts: list[str]) -> list[str]:
        if not texts:
            return []
        try:
            inputs = bundle.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            if self.torch is not None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad() if self.torch is not None else _nullcontext():
                generated = bundle.model.generate(**inputs, num_beams=4, max_new_tokens=256)
            return [bundle.tokenizer.decode(g, skip_special_tokens=True) for g in generated]
        except Exception as exc:
            logger.warning("Translation with Marian bundle failed: {}", exc)
            return texts

    def gpu_info(self) -> dict:
        if self.device != "cuda" or self.torch is None:
            return {"device": "cpu"}
        free, total = self.torch.cuda.mem_get_info()
        return {"device": "cuda", "vram_total": total, "vram_free": free}

    def cleanup(self) -> None:
        self._manga_ocr = None
        self._easyocr = None
        self._translator_ja_en = None
        self._translator_ko_en = None
        self._m2m = None
        self._m2m_tok = None
        self._inpaint_pipe = None
        self._img2img_pipe = None
        self._quality_img2img_pipe = None
        self._controlnet_img2img = None
        self._controlnet_modes: list[str] = []
        self._quality_controlnet_img2img = None
        self._quality_controlnet_modes: list[str] = []
        self._llm = None
        self._capability_cache.clear()
        self._inpaint_load_failed = False
        self._img2img_load_failed = False
        self._quality_img2img_load_failed = False
        self._controlnet_load_failed = False
        self._quality_controlnet_load_failed = False
        if self.device == "cuda" and self.torch is not None:
            self.torch.cuda.empty_cache()

    def warmup(
        self,
        include_ocr: bool = True,
        include_translation: bool = True,
        include_diffusion: bool = False,
        include_inpaint: bool = True,
        include_llm: bool = False,
        require_ocr: bool = True,
        require_translation: bool = True,
        require_diffusion: bool = False,
        require_inpaint: bool = False,
    ) -> dict:
        start = time.perf_counter()
        steps: list[dict[str, Any]] = []

        def _run_step(
            name: str,
            requested: bool,
            required: bool,
            loader: Callable[[], Any],
            success_check: Callable[[Any], bool],
        ) -> tuple[bool, Optional[str]]:
            if not requested:
                steps.append(
                    {
                        "name": name,
                        "requested": False,
                        "required": bool(required),
                        "loaded": False,
                        "ok": not required,
                        "duration_ms": 0,
                        "error": "",
                    }
                )
                return (not required), None

            t0 = time.perf_counter()
            error = ""
            loaded = False
            try:
                resource = loader()
                loaded = bool(success_check(resource))
            except Exception as exc:  # pragma: no cover - defensive path
                error = str(exc)
                loaded = False
            duration_ms = int((time.perf_counter() - t0) * 1000)
            ok = bool(loaded or not required)
            steps.append(
                {
                    "name": name,
                    "requested": True,
                    "required": bool(required),
                    "loaded": bool(loaded),
                    "ok": ok,
                    "duration_ms": duration_ms,
                    "error": error,
                }
            )
            return ok, (error or None)

        def _translation_loader():
            ja = self.load_ja_en()
            ko = self.load_ko_en()
            m2m_tok, m2m_model = self.load_m2m100()
            return {"ja": ja, "ko": ko, "m2m": bool(m2m_tok is not None and m2m_model is not None)}

        def _translation_ok(payload: dict[str, Any]) -> bool:
            return bool(payload.get("ja") is not None or payload.get("ko") is not None or payload.get("m2m"))

        _run_step(
            name="ocr",
            requested=include_ocr,
            required=require_ocr,
            loader=lambda: {"manga_ocr": self.load_manga_ocr(), "easyocr": self.load_easyocr()},
            success_check=lambda payload: bool(
                payload.get("manga_ocr") is not None or payload.get("easyocr") is not None
            ),
        )
        _run_step(
            name="translation",
            requested=include_translation,
            required=require_translation,
            loader=_translation_loader,
            success_check=_translation_ok,
        )
        _run_step(
            name="inpaint",
            requested=include_inpaint,
            required=require_inpaint,
            loader=self.load_inpaint_pipeline,
            success_check=lambda resource: resource is not None,
        )
        _run_step(
            name="img2img",
            requested=include_diffusion,
            required=require_diffusion,
            loader=self.load_img2img_pipeline,
            success_check=lambda resource: resource is not None,
        )
        _run_step(
            name="llm_post_edit",
            requested=include_llm,
            required=False,
            loader=self.load_llm,
            success_check=lambda resource: resource is not None,
        )

        ok = all(bool(step.get("ok", False)) for step in steps)
        required_missing = [
            str(step.get("name"))
            for step in steps
            if bool(step.get("required", False)) and not bool(step.get("loaded", False))
        ]
        return {
            "ok": ok,
            "device": self.device,
            "total_ms": int((time.perf_counter() - start) * 1000),
            "required_missing": required_missing,
            "steps": steps,
            "capabilities": self.capabilities(refresh=True),
        }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
