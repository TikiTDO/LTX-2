import logging
from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.device_map import resolve_transformer_device_map_preset
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    try_offload_to_cpu,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.telemetry import Timer, log_cuda_memory, log_nvidia_smi, log_ram_memory, log_system_summary
from ltx_pipelines.utils.text_context import TextContexts, load_text_contexts, save_text_contexts
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        temporal_upsampler_path: str | None,
        gemma_root: str | None,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = device,
        quantization: QuantizationPolicy | None = None,
        transformer_device_map: dict[str, str] | None = None,
        transformer_offload_dir: str | None = None,
        text_encoder_backend: str = "gemma-hf",
        gemma_device_map: str = "",
        gemma_move_vision_tower_to: str = "",
        temporal_upsample: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            temporal_upsampler_path=temporal_upsampler_path,
            loras=loras,
            quantization=quantization,
            transformer_device_map=transformer_device_map,
            transformer_offload_dir=transformer_offload_dir,
            text_encoder_backend=text_encoder_backend,
            gemma_device_map=gemma_device_map,
            gemma_move_vision_tower_to=gemma_move_vision_tower_to,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.temporal_upsample = bool(temporal_upsample)

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        context_in_path: str | None = None,
        context_out_path: str | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        log_cuda_memory("two_stage:begin")
        log_ram_memory("two_stage:begin")
        log_nvidia_smi("two_stage:begin")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        if context_in_path is not None:
            if enhance_prompt:
                raise ValueError("enhance_prompt requires a live text encoder (disable it when using context_in_path)")
            t = Timer("text_contexts_load")
            contexts = load_text_contexts(context_in_path, device=self.device)
            v_context_p, a_context_p, v_context_n, a_context_n = (
                contexts.v_context_p,
                contexts.a_context_p,
                contexts.v_context_n,
                contexts.a_context_n,
            )
            t.done()
        else:
            t = Timer("text_encoder_load+encode")
            text_encoder = self.stage_1_model_ledger.text_encoder()
            if enhance_prompt:
                prompt = generate_enhanced_prompt(
                    text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
                )
            context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
            v_context_p, a_context_p = context_p
            v_context_n, a_context_n = context_n
            # If the text encoder is CPU-backed (e.g. gemma-hf-cpu), move contexts to the pipeline device.
            v_context_p = v_context_p.to(self.device)
            a_context_p = a_context_p.to(self.device)
            v_context_n = v_context_n.to(self.device)
            a_context_n = a_context_n.to(self.device)
            t.done()

            if context_out_path is not None:
                save_text_contexts(
                    context_out_path,
                    TextContexts(
                        v_context_p=v_context_p,
                        a_context_p=a_context_p,
                        v_context_n=v_context_n,
                        a_context_n=a_context_n,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                    ),
                )

            # Offload text encoder ASAP to free VRAM before loading the diffusion transformer.
            try_offload_to_cpu(text_encoder)
            del text_encoder
            cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        t = Timer("stage1_models_load")
        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        t.done()
        log_cuda_memory("two_stage:after_stage1_models_load")
        log_ram_memory("two_stage:after_stage1_models_load")
        log_nvidia_smi("two_stage:after_stage1_models_load")
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()
        log_cuda_memory("two_stage:after_stage1_denoise")
        log_ram_memory("two_stage:after_stage1_denoise")
        log_nvidia_smi("two_stage:after_stage1_denoise")

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        t = Timer("stage2_upsample")
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )
        t.done()

        torch.cuda.synchronize()
        cleanup_memory()

        t = Timer("stage2_transformer_load")
        transformer = self.stage_2_model_ledger.transformer()
        t.done()
        log_cuda_memory("two_stage:after_stage2_transformer_load")
        log_ram_memory("two_stage:after_stage2_transformer_load")
        log_nvidia_smi("two_stage:after_stage2_transformer_load")
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()
        log_cuda_memory("two_stage:after_stage2_denoise")
        log_ram_memory("two_stage:after_stage2_denoise")
        log_nvidia_smi("two_stage:after_stage2_denoise")

        t = Timer("decode_video+audio")
        if self.temporal_upsample:
            video_state = replace(
                video_state,
                latent=upsample_video(
                    latent=video_state.latent[:1],
                    video_encoder=video_encoder,
                    upsampler=self.stage_2_model_ledger.temporal_upsampler(),
                ),
            )
        decoded_video = vae_decode_video(
            video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
        )
        t.done()
        del video_encoder

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log_system_summary("ti2vid_two_stages")
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    transformer_device_map = resolve_transformer_device_map_preset(args.transformer_device_map) if args.dispatch_transformer else None
    transformer_offload_dir = args.transformer_offload_dir or None
    if not args.gemma_root:
        raise ValueError("--gemma-root is required for --text-encoder gemma-hf/gemma-bnb4")
    gemma_root = args.gemma_root or None
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=args.lora,
        quantization=args.quantization,
        transformer_device_map=dict(transformer_device_map) if transformer_device_map is not None else None,
        transformer_offload_dir=transformer_offload_dir,
        text_encoder_backend=args.text_encoder,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
