from dataclasses import replace
import logging
import os

import torch

from ltx_core.loader import SDOps
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.encoders.av_encoder import GEMMA_MODEL_OPS
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.gemma_device_map import resolve_gemma_layer_device_map_preset
from ltx_pipelines.utils.gemma_shard import shard_gemma3_language_layers_inplace

logger = logging.getLogger(__name__)

AV_GEMMA_TEXT_ENCODER_LIGHT_OPS = (
    SDOps("AV_GEMMA_TEXT_ENCODER_LIGHT_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
)


class ModelLedger:
    """
    Central coordinator for loading and building models used in an LTX pipeline.
    The ledger wires together multiple model builders (transformer, video VAE encoder/decoder,
    audio VAE decoder, vocoder, text encoder, and optional latent upsampler) and exposes
    factory methods for constructing model instances.
    ### Model Building
    Each model method (e.g. :meth:`transformer`, :meth:`video_decoder`, :meth:`text_encoder`)
    constructs a new model instance on each call. The builder uses the
    :class:`~ltx_core.loader.registry.Registry` to load weights from the checkpoint,
    instantiates the model with the configured ``dtype``, and moves it to ``self.device``.
    .. note::
        Models are **not cached**. Each call to a model method creates a new instance.
        Callers are responsible for storing references to models they wish to reuse
        and for freeing GPU memory (e.g. by deleting references and calling
        ``torch.cuda.empty_cache()``).
    ### Constructor parameters
    dtype:
        Torch dtype used when constructing all models (e.g. ``torch.bfloat16``).
    device:
        Target device to which models are moved after construction (e.g. ``torch.device("cuda")``).
    checkpoint_path:
        Path to a checkpoint directory or file containing the core model weights
        (transformer, video VAE, audio VAE, text encoder, vocoder). If ``None``, the
        corresponding builders are not created and calling those methods will raise
        a :class:`ValueError`.
    gemma_root_path:
        Base path to Gemma-compatible CLIP/text encoder weights. Required to
        initialize the text encoder builder; if omitted, :meth:`text_encoder` cannot be used.
    spatial_upsampler_path:
        Optional path to a latent upsampler checkpoint. If provided, the
        :meth:`spatial_upsampler` method becomes available; otherwise calling it raises
        a :class:`ValueError`.
    loras:
        Optional collection of LoRA configurations (paths, strengths, and key operations)
        that are applied on top of the base transformer weights when building the model.
    registry:
        Optional :class:`Registry` instance for weight caching across builders.
        Defaults to :class:`DummyRegistry` which performs no cross-builder caching.
    quantization:
        Optional :class:`QuantizationPolicy` controlling how transformer weights
        are stored and how matmul is executed. Defaults to None, which means no quantization.
    ### Creating Variants
    Use :meth:`with_loras` to create a new ``ModelLedger`` instance that includes
    additional LoRA configurations while sharing the same registry for weight caching.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        temporal_upsampler_path: str | None = None,
        loras: LoraPathStrengthAndSDOps | None = None,
        registry: Registry | None = None,
        quantization: QuantizationPolicy | None = None,
        transformer_device_map: dict[str, str] | None = None,
        transformer_offload_dir: str | None = None,
        text_encoder_backend: str = "gemma-hf",
        gemma_device_map: str = "",
        gemma_move_vision_tower_to: str = "",
    ):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.temporal_upsampler_path = temporal_upsampler_path
        self.loras = loras or ()
        self.registry = registry or DummyRegistry()
        self.quantization = quantization
        self.transformer_device_map = transformer_device_map
        self.transformer_offload_dir = transformer_offload_dir
        self.text_encoder_backend = text_encoder_backend
        self.gemma_device_map = gemma_device_map
        self.gemma_move_vision_tower_to = gemma_move_vision_tower_to
        self.build_model_builders()

    def build_model_builders(self) -> None:
        if self.checkpoint_path is not None:
            self.transformer_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
            )

            self.vae_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vae_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vocoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            if self.gemma_root_path is not None:
                module_ops = module_ops_from_gemma_root(self.gemma_root_path)
                model_folder = find_matching_file(self.gemma_root_path, "model*.safetensors").parent
                weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

                self.text_encoder_builder = Builder(
                    model_path=(str(self.checkpoint_path), *weight_paths),
                    model_class_configurator=AVGemmaTextEncoderModelConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=self.registry,
                    module_ops=(GEMMA_MODEL_OPS, *module_ops),
                )

        if self.spatial_upsampler_path is not None:
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )

        if self.temporal_upsampler_path is not None:
            self.temporal_upsampler_builder = Builder(
                model_path=self.temporal_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry) or self.registry is None:
            return self.device
        else:
            return torch.device("cpu")

    def with_loras(self, loras: LoraPathStrengthAndSDOps) -> "ModelLedger":
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            temporal_upsampler_path=self.temporal_upsampler_path,
            loras=(*self.loras, *loras),
            registry=self.registry,
            quantization=self.quantization,
            transformer_device_map=self.transformer_device_map,
            transformer_offload_dir=self.transformer_offload_dir,
            text_encoder_backend=self.text_encoder_backend,
            gemma_device_map=self.gemma_device_map,
            gemma_move_vision_tower_to=self.gemma_move_vision_tower_to,
        )

    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        if self.transformer_device_map is not None:
            if self.quantization is not None:
                raise ValueError("Quantization + multi-device dispatch is not wired yet (keep it simple for now).")

            # Manual per-block dispatch to enable multi-GPU and CPU-RAM sharding without Accelerate hooks.
            # The LTXModel implementation is patched to move activations to each block's device during the forward pass.
            velocity_model = self.transformer_builder.build(device=torch.device("cpu"), dtype=self.dtype).eval()

            # Keep non-block modules on the primary device (inputs/outputs, preprocessors, etc.)
            for name, child in velocity_model.named_children():
                if name != "transformer_blocks":
                    child.to(self.device)

            # Dispatch blocks by index using keys like "transformer_blocks.0".
            default_block_device = self.device
            block_counts: dict[str, int] = {}

            def _rank_cuda_devices_by_free_mem() -> list[str]:
                if not torch.cuda.is_available():
                    return []
                ranked: list[tuple[int, str]] = []
                for i in range(torch.cuda.device_count()):
                    try:
                        free_b, _total_b = torch.cuda.mem_get_info(i)
                        ranked.append((int(free_b), f"cuda:{i}"))
                    except Exception:
                        ranked.append((0, f"cuda:{i}"))
                ranked.sort(reverse=True)
                return [dev for _free, dev in ranked]

            def _move_block_with_oom_fallback(block: torch.nn.Module, preferred: torch.device) -> torch.device:
                # If a target GPU runs out of memory during .to(), try other GPUs (most free first) then CPU.
                # Disable this behavior by setting LTX_STRICT_DEVICE_MAP=1.
                strict = (os.environ.get("LTX_STRICT_DEVICE_MAP", "") or "").strip() in ("1", "true", "TRUE", "yes")
                tried: list[str] = []

                def _try(dev: torch.device) -> torch.device | None:
                    tried.append(str(dev))
                    try:
                        block.to(dev)
                        return dev
                    except torch.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return None

                ok = _try(preferred)
                if ok is not None or strict:
                    if ok is None:
                        raise torch.OutOfMemoryError(f"OOM moving block to {preferred}; tried={tried!r}")
                    return ok

                # Try other CUDA devices first (if any), then CPU.
                for dev_str in _rank_cuda_devices_by_free_mem():
                    dev = torch.device(dev_str)
                    if str(dev) == str(preferred):
                        continue
                    ok = _try(dev)
                    if ok is not None:
                        logger.warning("transformer_block_oom_fallback from=%s to=%s tried=%s", preferred, dev, tried)
                        return ok

                ok = _try(torch.device("cpu"))
                if ok is not None:
                    logger.warning("transformer_block_oom_fallback from=%s to=%s tried=%s", preferred, "cpu", tried)
                    return ok
                raise torch.OutOfMemoryError(f"OOM moving block; preferred={preferred} tried={tried!r}")

            for idx, block in enumerate(getattr(velocity_model, "transformer_blocks")):
                key = f"transformer_blocks.{idx}"
                dev_str = self.transformer_device_map.get(key, None)
                preferred = default_block_device if dev_str is None else torch.device(dev_str)
                actual = _move_block_with_oom_fallback(block, preferred)
                block_counts[str(actual)] = block_counts.get(str(actual), 0) + 1

            # Lightweight visibility into where blocks ended up after OOM fallbacks.
            velocity_model._ltx_block_device_counts = block_counts  # type: ignore[attr-defined]
            logger.info("transformer_block_devices %s", " ".join(f"{k}={v}" for k, v in sorted(block_counts.items())))
            if self.transformer_offload_dir:
                logger.warning("transformer_offload_dir is set but is currently unused (manual sharding path)")

            return X0Model(velocity_model).eval()

        if self.quantization is None:
            return (
                X0Model(self.transformer_builder.build(device=self._target_device(), dtype=self.dtype))
                .to(self.device)
                .eval()
            )
        else:
            sd_ops = self.transformer_builder.model_sd_ops
            if self.quantization.sd_ops is not None:
                sd_ops = SDOps(
                    name=f"sd_ops_chain_{sd_ops.name}+{self.quantization.sd_ops.name}",
                    mapping=(*sd_ops.mapping, *self.quantization.sd_ops.mapping),
                )
            builder = replace(
                self.transformer_builder,
                module_ops=(*self.transformer_builder.module_ops, *self.quantization.module_ops),
                model_sd_ops=sd_ops,
            )
            return X0Model(builder.build(device=self._target_device())).to(self.device).eval()

    def video_decoder(self) -> VideoDecoder:
        if not hasattr(self, "vae_decoder_builder"):
            raise ValueError(
                "Video decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.vae_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def video_encoder(self) -> VideoEncoder:
        if not hasattr(self, "vae_encoder_builder"):
            raise ValueError(
                "Video encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.vae_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def text_encoder(self) -> AVGemmaTextEncoderModel:
        if self.text_encoder_backend == "gemma-bnb4":
            return self._text_encoder_gemma_bnb4().eval()

        force_cpu = self.text_encoder_backend == "gemma-hf-cpu"
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError(
                "Text encoder not initialized. Please provide a checkpoint path and gemma root path to the "
                "ModelLedger constructor."
            )

        # Always build on CPU to avoid CUDA OOM spikes during state_dict materialization.
        enc = self.text_encoder_builder.build(device=torch.device("cpu"), dtype=self.dtype).eval()
        if force_cpu:
            return enc.eval()

        # Manual multi-GPU sharding for the underlying HF Gemma3 model, avoiding Accelerate.
        if torch.cuda.is_available() and self.device.type == "cuda":
            preset = (self.gemma_device_map or "").strip()
            if preset == "" and torch.cuda.device_count() >= 2:
                preset = "middle-2gpu"
            layer_map = resolve_gemma_layer_device_map_preset(preset) if preset else None
            if layer_map is not None:
                move_vision_to = None
                if (self.gemma_move_vision_tower_to or "").strip():
                    move_vision_to = torch.device(self.gemma_move_vision_tower_to)
                shard_gemma3_language_layers_inplace(
                    enc.model,
                    layer_device_map=layer_map,
                    primary=torch.device("cuda:0"),
                    move_vision_tower_to=move_vision_to,
                )
                # Keep the lightweight projection/connector modules next to the tied embedding/lm_head.
                enc.feature_extractor_linear = enc.feature_extractor_linear.to(torch.device("cuda:0"))
                enc.embeddings_connector = enc.embeddings_connector.to(torch.device("cuda:0"))
                enc.audio_embeddings_connector = enc.audio_embeddings_connector.to(torch.device("cuda:0"))
                return enc.eval()

        # Single-GPU fallback: move entire text encoder to pipeline device (may OOM for large Gemma).
        return enc.to(self.device).eval()

    def _text_encoder_gemma_bnb4(self) -> AVGemmaTextEncoderModel:
        if self.checkpoint_path is None or self.gemma_root_path is None:
            raise ValueError("gemma-bnb4 text encoder requires checkpoint_path and gemma_root_path")

        try:
            from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration, Gemma3Processor, AutoImageProcessor
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers is required for gemma-bnb4 text encoder") from e

        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError("bitsandbytes is required for --text-encoder gemma-bnb4") from e

        # Build the wrapper module (feature extractor + connectors) from the LTX checkpoint config.
        cfg = Builder(
            model_path=self.checkpoint_path,
            model_class_configurator=AVGemmaTextEncoderModelConfigurator,
            model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            registry=self.registry,
        ).model_config()
        wrapper = AVGemmaTextEncoderModelConfigurator.from_config(cfg)
        wrapper = wrapper.to(dtype=self.dtype)

        # Load only LTX-specific text-encoder weights (projection + connectors) from the checkpoint.
        loader = SafetensorsModelStateDictLoader()
        state_dict = loader.load(self.checkpoint_path, sd_ops=AV_GEMMA_TEXT_ENCODER_LIGHT_OPS, device=torch.device("cpu"))
        wrapper.load_state_dict(state_dict.sd, strict=False, assign=True)

        # Tokenizer/processor roots (same logic as module_ops_from_gemma_root).
        tokenizer_root = str(find_matching_file(self.gemma_root_path, "tokenizer.model").parent)
        processor_root = str(find_matching_file(self.gemma_root_path, "preprocessor_config.json").parent)
        module_ops = module_ops_from_gemma_root(self.gemma_root_path)
        for op in module_ops:
            if op.matcher(wrapper):
                wrapper = op.mutator(wrapper)

        # Load Gemma weights via HF + bitsandbytes (4-bit), for prompt encoding only.
        #
        # Important: Gemma3 + bitsandbytes quantization has been flaky when `from_pretrained` triggers
        # Accelerate device_map dispatch + meta initialization, yielding:
        #   ValueError: weight is on the meta device ...
        # Force a non-meta load path and keep device placement explicit.
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to(self.device)
        wrapper.model = gemma

        # Ensure processor is present (enhance_prompt uses it).
        if wrapper.processor is None and wrapper.tokenizer is not None:
            img_proc = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)
            wrapper.processor = Gemma3Processor(image_processor=img_proc, tokenizer=wrapper.tokenizer.tokenizer)

        # Populate rope buffers expected by the custom key mapping.
        wrapper = GEMMA_MODEL_OPS.mutator(wrapper)

        # Keep the lightweight parts next to the Gemma model.
        wrapper.feature_extractor_linear = wrapper.feature_extractor_linear.to(self.device)
        wrapper.embeddings_connector = wrapper.embeddings_connector.to(self.device)
        wrapper.audio_embeddings_connector = wrapper.audio_embeddings_connector.to(self.device)
        return wrapper

    def audio_decoder(self) -> AudioDecoder:
        if not hasattr(self, "audio_decoder_builder"):
            raise ValueError(
                "Audio decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.audio_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def vocoder(self) -> Vocoder:
        if not hasattr(self, "vocoder_builder"):
            raise ValueError(
                "Vocoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.vocoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def spatial_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "upsampler_builder"):
            raise ValueError("Upsampler not initialized. Please provide upsampler path to the ModelLedger constructor.")

        return self.upsampler_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def temporal_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "temporal_upsampler_builder"):
            raise ValueError(
                "Temporal upsampler not initialized. Please provide temporal_upsampler_path to the ModelLedger constructor."
            )
        return (
            self.temporal_upsampler_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        )
