import os
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import MILLISECONDS_TO_SECONDS_CONVERSION, TERM_PLOTLIB_AVAILABLE, BenchmarkMetrics, TaskType
from vllm.tokenizers import TokenizerLike

from vllm_omni.metrics import definitions as defs


@dataclass
class MultiModalsBenchmarkMetrics(BenchmarkMetrics):
    mean_audio_ttfp_ms: float = 0.0
    median_audio_ttfp_ms: float = 0.0
    std_audio_ttfp_ms: float = 0.0
    percentiles_audio_ttfp_ms: list[tuple[float, float]] = None
    total_audio_duration_s: float = 0.0
    total_audio_frames: int = 0
    audio_throughput: float = 0.0
    mean_audio_rtf: float = 0.0
    median_audio_rtf: float = 0.0
    std_audio_rtf: float = 0.0
    percentiles_audio_rtf: list[tuple[float, float]] = None
    mean_audio_duration_s: float = 0.0
    median_audio_duration_s: float = 0.0
    std_audio_duration_s: float = 0.0
    percentiles_audio_duration_s: list[tuple[float, float]] = None
    # Streaming-audio continuity (see vllm_omni/benchmarks/audio_continuity.py).
    mean_audio_underrun_s: float = 0.0
    median_audio_underrun_s: float = 0.0
    std_audio_underrun_s: float = 0.0
    percentiles_audio_underrun_s: list[tuple[float, float]] = None
    audio_continuity_ok_rate: float = 1.0
    total_images: int = 0
    image_throughput: float = 0.0
    average_pixels_per_image: float = 0.0
    mean_denoise_step_latency_ms: float = 0.0
    mean_image_generation_ms: float = 0.0
    median_image_generation_ms: float = 0.0
    std_image_generation_ms: float = 0.0
    percentiles_image_generation_ms: list[tuple[float, float]] = None


@dataclass
class StageBenchmarkMetrics:
    """Aggregated metrics for one pipeline stage (for printing only)."""

    stage_id: int
    stage_name: str
    final_output_type: str
    total_output: int
    ttfts: list[float]
    tpots: list[float]
    itls: list[float]
    vllm_ttfts: list[float]
    vllm_tpots: list[float]
    vllm_itls: list[float]
    audio_ttfps: list[float]
    audio_rtfs: list[float]
    audio_durations: list[float]
    audio_frames: list[int]
    missing_audio_duration_count: int
    missing_audio_rtf_count: int
    stage_gen_times_ms: list[float]
    postprocess_times_ms: list[float]
    output_unit_type: str
    output_unit_count: int
    serving_time_to_first_outputs_ms: list[float]
    time_per_output_units_ms: list[float]
    inter_output_latencies_ms: list[float]


def _percentile_rows_seconds(
    values: list[float], selected_percentiles: list[float], *, to_ms: bool
) -> tuple[float, float, list[tuple[float, float]]]:
    arr = np.asarray(values, dtype=float)
    scale = 1000.0 if to_ms else 1.0
    mean_v = float(np.mean(arr)) * scale
    median_v = float(np.median(arr)) * scale
    rows = [(float(p), float(np.percentile(arr, p)) * scale) for p in selected_percentiles]
    return mean_v, median_v, rows


def _p_label(p: float) -> str:
    pf = float(p)
    if abs(pf - round(pf)) < 1e-9:
        return str(int(round(pf)))
    return str(pf)


_STREAMING_OUTPUT_UNIT_TYPES = frozenset(
    {
        "text",
        "stream",
        "audio",
    }
)


def _wants_text_ttft(metrics: list[str]) -> bool:
    return "ttft" in metrics


def _wants_text_tpot(metrics: list[str]) -> bool:
    return "tpot" in metrics or "tpop" in metrics


def _wants_text_itl(metrics: list[str]) -> bool:
    return "itl" in metrics


def _wants_stream_ttfc(metrics: list[str]) -> bool:
    return "ttfc" in metrics


def _wants_stream_tpoc(metrics: list[str]) -> bool:
    return "tpoc" in metrics or "tpop" in metrics


def _wants_stream_icl(metrics: list[str]) -> bool:
    return "icl" in metrics


def _should_print_stage_metrics() -> bool:
    return os.environ.get("VLLM_OMNI_PRINT_STAGE") == "1"


def _stage_modality_flags(
    final_output_type: str,
    output_unit_type: str,
) -> tuple[bool, bool, bool, bool, bool]:
    is_text_stage = final_output_type == "text" or output_unit_type == "text"
    is_audio_stage = final_output_type == "audio" or output_unit_type == "audio"
    is_image_stage = final_output_type in {"image", "images"} or output_unit_type == "image"
    is_video_stage = final_output_type in {"video", "videos"} or output_unit_type == "video"
    is_internal_stream_stage = (
        output_unit_type in _STREAMING_OUTPUT_UNIT_TYPES and not is_text_stage and not is_audio_stage
    )
    return (
        is_text_stage,
        is_audio_stage,
        is_image_stage,
        is_video_stage,
        is_internal_stream_stage,
    )


def print_metrics(
    task_type,
    selected_percentile_metrics,
    max_concurrency,
    request_rate,
    benchmark_duration,
    goodput_config_dict,
    metrics: MultiModalsBenchmarkMetrics,
    *,
    outputs: list[RequestFuncOutput] | None = None,
    selected_percentiles: list[float] | None = None,
):
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        print("{:<40} {:<10.2f}".format("Peak concurrent requests:", metrics.max_concurrent_requests))
    if task_type != TaskType.GENERATION or "e2el" in selected_percentile_metrics:
        process_one_metric("e2el", metrics)
    print_text_metrics(task_type, selected_percentile_metrics, metrics)
    if task_type == TaskType.GENERATION:
        if _has_audio_output(metrics):
            print_audio_metrics(selected_percentile_metrics, metrics)
        if _has_image_output(metrics):
            print_image_metrics(selected_percentiles or [], metrics)
        if _should_print_stage_metrics() and outputs and selected_percentiles is not None:
            print("\n{s:{c}^{n}}".format(s=" Stage Benchmark Result ", n=50, c="="))
            for sm in _build_stage_metrics_from_outputs(outputs):
                print_stage_metrics(
                    task_type,
                    selected_percentile_metrics,
                    selected_percentiles,
                    sm,
                )
    print("=" * 50)


def print_text_metrics(task_type, selected_percentile_metrics, metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Text Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
        print("{:<40} {:<10.2f}".format("Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s))
        print("{:<40} {:<10.2f}".format("Peak concurrent requests:", metrics.max_concurrent_requests))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    if task_type == TaskType.GENERATION:
        # No text tokens generated (e.g. pure TTS speech endpoint): per-token
        # latency metrics (ttft/tpot/itl) are undefined, so skip them.
        has_text_output = metrics.total_output > 0
        if has_text_output:
            if _wants_text_ttft(selected_percentile_metrics):
                process_one_metric("ttft", metrics)
            if _wants_text_tpot(selected_percentile_metrics):
                process_one_metric("tpot", metrics)
            if _wants_text_itl(selected_percentile_metrics):
                process_one_metric("itl", metrics)


def _has_audio_output(metrics: MultiModalsBenchmarkMetrics) -> bool:
    return bool(getattr(metrics, "total_audio_duration_s", 0.0) > 0 or getattr(metrics, "total_audio_frames", 0) > 0)


def _has_image_output(metrics: MultiModalsBenchmarkMetrics) -> bool:
    return int(getattr(metrics, "total_images", 0) or 0) > 0


def print_audio_metrics(selected_percentile_metrics, metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Audio Result ", n=50, c="="))
    print("{:<40} {:<10.2f}".format("Total audio duration generated(s):", metrics.total_audio_duration_s))
    print("{:<40} {:<10}".format("Total audio frames generated:", metrics.total_audio_frames))
    print("{:<40} {:<10.2f}".format("Audio throughput(audio duration/s):", metrics.audio_throughput))
    print(
        "{:<40} {:<10.2%}".format(
            "Streaming continuity OK rate:",
            metrics.audio_continuity_ok_rate,
        )
    )
    for metric in selected_percentile_metrics:
        if metric.startswith("audio"):
            process_one_metric(metric, metrics)


def print_image_metrics(selected_percentiles: list[float], metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Image Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total images generated:", metrics.total_images))
    print("{:<40} {:<10.2f}".format("Image throughput (img/s):", metrics.image_throughput))
    print("{:<40} {:<10.2f}".format("Average pixels per image:", metrics.average_pixels_per_image))
    print("{:<40} {:<10.2f}".format("Mean denoise step latency (ms):", metrics.mean_denoise_step_latency_ms))
    if metrics.mean_image_generation_ms > 0:
        print("-----------------Image Generation-----------------")
        print("{:<40} {:<10.2f}".format("Mean IMAGE_GENERATION (ms):", metrics.mean_image_generation_ms))
        print("{:<40} {:<10.2f}".format("Median IMAGE_GENERATION (ms):", metrics.median_image_generation_ms))
        for p, value in metrics.percentiles_image_generation_ms or []:
            if p in selected_percentiles:
                print("{:<40} {:<10.2f}".format(f"P{_p_label(p)} IMAGE_GENERATION (ms):", value))


def process_one_metric(
    metric_attribute_name: str,
    metrics: MultiModalsBenchmarkMetrics,
):
    metric_header_map = {
        "ttft": "Time to First Token",
        "tpot": "Time per Output Token (excl. 1st token)",
        "itl": "Inter-token Latency",
        "e2el": "End-to-end Latency",
        defs.AUDIO_TTFP: "Time to First Packet",
        defs.AUDIO_RTF: "Real Time Factor",
        defs.AUDIO_DURATION: "Audio Duration",
        defs.AUDIO_UNDERRUN: "Streaming Audio Underrun",
    }

    header = metric_header_map.get(metric_attribute_name, metric_attribute_name)
    print("{s:{c}^{n}}".format(s=header, n=50, c="-"))

    is_audio_rtf = metric_attribute_name == defs.AUDIO_RTF
    is_audio_duration_or_underrun = metric_attribute_name in (defs.AUDIO_DURATION, defs.AUDIO_UNDERRUN)

    suffix = "_ms"
    unit_suffix = " (ms)"
    if is_audio_duration_or_underrun:
        suffix = "_s"
        unit_suffix = " (s)"
    elif is_audio_rtf:
        suffix = ""
        unit_suffix = ""
    mean_attr_name = f"mean_{metric_attribute_name}{suffix}"
    mean_value = getattr(metrics, mean_attr_name, 0.0)
    print(f"{f'Mean {metric_attribute_name.upper()}{unit_suffix}:':<40} {mean_value:<10.2f}")

    median_attr_name = f"median_{metric_attribute_name}{suffix}"
    median_value = getattr(metrics, median_attr_name, 0.0)
    print(f"{f'Median {metric_attribute_name.upper()}{unit_suffix}:':<40} {median_value:<10.2f}")

    percentiles_attr_name = f"percentiles_{metric_attribute_name}{suffix}"
    percentiles = getattr(metrics, percentiles_attr_name, [])

    for percentile, value in percentiles:
        p_str = str(int(percentile)) if percentile.is_integer() else str(percentile)
        label = f"P{p_str} {metric_attribute_name.upper()}{unit_suffix}:"
        print(f"{label:<40} {value:<10.2f}")


def _print_percentile_metric(
    header: str,
    label: str,
    values: list[float],
    selected_percentiles: list[float],
    *,
    values_are_ms: bool = False,
    to_ms: bool = True,
) -> None:
    if not values:
        return
    values_s = [v / 1000.0 for v in values] if values_are_ms else values
    mean_v, med_v, pcts = _percentile_rows_seconds(values_s, selected_percentiles, to_ms=to_ms)
    unit_suffix = " (ms)" if to_ms else ""
    print("{s:{c}^{n}}".format(s=header, n=50, c="-"))
    print(f"{f'Mean {label}{unit_suffix}:':<40} {mean_v:<10.2f}")
    print(f"{f'Median {label}{unit_suffix}:':<40} {med_v:<10.2f}")
    for p, v in pcts:
        p_str = _p_label(p)
        print(f"{f'P{p_str} {label}{unit_suffix}:':<40} {v:<10.2f}")


def _print_stage_timing(sm: StageBenchmarkMetrics, selected_percentiles: list[float]) -> None:
    _print_percentile_metric(
        "Stage Timing",
        "stage_gen_time",
        sm.stage_gen_times_ms,
        selected_percentiles,
        values_are_ms=True,
        to_ms=True,
    )
    if sm.postprocess_times_ms and any(v > 0 for v in sm.postprocess_times_ms):
        _print_percentile_metric(
            "Stage Postprocess",
            "postprocess",
            sm.postprocess_times_ms,
            selected_percentiles,
            values_are_ms=True,
            to_ms=True,
        )


def _print_text_stage_metrics(
    task_type: TaskType,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    sm: StageBenchmarkMetrics,
) -> None:
    print("{s:{c}^{n}}".format(s=" Text Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Stage generated tokens:", sm.total_output))

    if task_type != TaskType.GENERATION:
        return
    if _wants_text_ttft(selected_percentile_metrics):
        _print_percentile_metric("Serving Time to First Token", "Serving TTFT", sm.ttfts, selected_percentiles)
        _print_percentile_metric("Time to First Token", "TTFT", sm.vllm_ttfts, selected_percentiles)
    if _wants_text_tpot(selected_percentile_metrics):
        _print_percentile_metric(
            "Time per Output Token (excl. 1st token)",
            "TPOT",
            sm.vllm_tpots,
            selected_percentiles,
        )
    if _wants_text_itl(selected_percentile_metrics):
        _print_percentile_metric("Inter-token Latency", "ITL", sm.vllm_itls, selected_percentiles)


def _print_audio_stage_metrics(
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    sm: StageBenchmarkMetrics,
) -> None:
    total_duration = float(sum(sm.audio_durations))
    total_frames = int(sum(sm.audio_frames))
    print("{s:{c}^{n}}".format(s=" Audio Result ", n=50, c="="))
    print("{:<40} {:<10.2f}".format("Stage audio duration generated(s):", total_duration))
    print("{:<40} {:<10}".format("Stage audio frames generated:", total_frames))
    if defs.AUDIO_TTFP in selected_percentile_metrics:
        _print_percentile_metric(
            "Serving Time to First Packet", "Serving AUDIO_TTFP", sm.audio_ttfps, selected_percentiles
        )
    if defs.AUDIO_RTF in selected_percentile_metrics:
        if sm.audio_rtfs:
            _print_percentile_metric("Real Time Factor", "AUDIO_RTF", sm.audio_rtfs, selected_percentiles, to_ms=False)
        elif sm.missing_audio_rtf_count > 0:
            print("{:<40} {:<10}".format("AUDIO_RTF skipped:", "missing stage-local audio duration/sample rate"))
    if defs.AUDIO_DURATION in selected_percentile_metrics:
        if sm.audio_durations:
            _print_percentile_metric(
                "Audio Duration",
                "AUDIO_DURATION",
                sm.audio_durations,
                selected_percentiles,
                to_ms=False,
            )
        elif sm.missing_audio_duration_count > 0:
            print("{:<40} {:<10}".format("AUDIO_DURATION skipped:", "missing stage-local audio duration/sample rate"))


def _print_image_stage_metrics(
    selected_percentiles: list[float],
    sm: StageBenchmarkMetrics,
) -> None:
    image_num = sm.output_unit_count if sm.output_unit_type == "image" else 0
    print("{s:{c}^{n}}".format(s=" Image Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total images generated:", image_num))
    _print_percentile_metric(
        "Image Generation",
        "IMAGE_GENERATION",
        sm.stage_gen_times_ms,
        selected_percentiles,
        values_are_ms=True,
        to_ms=True,
    )


def _print_internal_stream_stage_metrics(
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    sm: StageBenchmarkMetrics,
) -> None:
    print("{s:{c}^{n}}".format(s=" Internal Stream Result ", n=50, c="="))
    if _wants_stream_ttfc(selected_percentile_metrics) and sm.serving_time_to_first_outputs_ms:
        _print_percentile_metric(
            "Serving Time to First Chunk",
            "Serving TTFC",
            sm.serving_time_to_first_outputs_ms,
            selected_percentiles,
            values_are_ms=True,
            to_ms=True,
        )
    if _wants_stream_tpoc(selected_percentile_metrics) and sm.time_per_output_units_ms:
        _print_percentile_metric(
            "Time per Output Chunk (excl. 1st chunk)",
            "TPOP",
            sm.time_per_output_units_ms,
            selected_percentiles,
            values_are_ms=True,
            to_ms=True,
        )
    if _wants_stream_icl(selected_percentile_metrics) and sm.inter_output_latencies_ms:
        _print_percentile_metric(
            "Inter-chunk Latency",
            "ICL",
            sm.inter_output_latencies_ms,
            selected_percentiles,
            values_are_ms=True,
            to_ms=True,
        )


def print_stage_metrics(
    task_type: TaskType,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    sm: StageBenchmarkMetrics,
):
    title = f" Stage {sm.stage_id} ({sm.stage_name}) "
    (
        is_text_stage,
        is_audio_stage,
        is_image_stage,
        is_video_stage,
        is_internal_stream_stage,
    ) = _stage_modality_flags(sm.final_output_type, sm.output_unit_type)

    print("{s:{c}^{n}}".format(s=title, n=50, c="="))
    if is_image_stage:
        _print_image_stage_metrics(selected_percentiles, sm)
        return

    if is_video_stage:
        return

    _print_stage_timing(sm, selected_percentiles)
    if is_text_stage:
        _print_text_stage_metrics(task_type, selected_percentile_metrics, selected_percentiles, sm)
    elif is_audio_stage:
        _print_audio_stage_metrics(
            selected_percentile_metrics,
            selected_percentiles,
            sm,
        )
    elif is_internal_stream_stage:
        _print_internal_stream_stage_metrics(selected_percentile_metrics, selected_percentiles, sm)


def _build_stage_metrics_from_outputs(
    outputs: list[RequestFuncOutput],
) -> list[StageBenchmarkMetrics]:
    """Aggregate per ``stage_id`` using ``stage_metrics`` snapshots from the client."""
    buckets: dict[str, list[tuple[RequestFuncOutput, dict]]] = defaultdict(list)
    for out in outputs:
        if not getattr(out, "success", False):
            continue
        smap = getattr(out, "stage_metrics", None) or {}
        if not isinstance(smap, dict) or not smap:
            continue
        for sid, info in smap.items():
            if isinstance(info, dict):
                buckets[str(sid)].append((out, info))

    result: list[StageBenchmarkMetrics] = []
    for sid in sorted(buckets.keys(), key=lambda x: int(x)):
        rows = buckets[sid]
        stage_id_int = int(sid)
        stage_name = str((rows[0][1] or {}).get("stage_name") or f"stage_{stage_id_int}")
        final_output_type = str((rows[0][1] or {}).get("final_output_type") or "unknown")
        output_unit_types = [
            str((info or {}).get("output_unit_type") or "") for _, info in rows if (info or {}).get("output_unit_type")
        ]
        output_unit_type = output_unit_types[0] if output_unit_types else "other"

        is_text_stage, is_audio_stage, _, _, is_internal_stream_stage = _stage_modality_flags(
            final_output_type, output_unit_type
        )
        total_output = 0

        ttfts: list[float] = []
        tpots: list[float] = []
        itls: list[float] = []
        vllm_ttfts: list[float] = []
        vllm_tpots: list[float] = []
        vllm_itls: list[float] = []
        stage_gen_times_ms = [float((info or {}).get("stage_gen_time_ms") or 0.0) for _, info in rows]
        postprocess_times_ms = [float((info or {}).get("postprocess_time_ms") or 0.0) for _, info in rows]
        output_unit_count = sum(int((info or {}).get("output_unit_count") or 0) for _, info in rows)
        inter_output_latencies_ms: list[float] = []
        serving_time_to_first_outputs_ms: list[float] = []
        time_per_output_units_ms: list[float] = []
        if is_text_stage or is_internal_stream_stage:
            serving_time_to_first_outputs_ms = [
                float((info or {}).get("serving_time_to_first_output_ms") or 0.0)
                for _, info in rows
                if (info or {}).get("serving_time_to_first_output_ms") is not None
            ]
            time_per_output_units_ms = [
                float((info or {}).get("time_per_output_unit_ms") or 0.0)
                for _, info in rows
                if int((info or {}).get("output_unit_count") or 0) > 1
                and (info or {}).get("time_per_output_unit_ms") is not None
            ]
            for _, info in rows:
                values = (info or {}).get("inter_output_latencies_ms")
                if isinstance(values, list):
                    inter_output_latencies_ms.extend(float(v or 0.0) for v in values)
                elif (info or {}).get("inter_output_latency_ms") is not None:
                    inter_output_latencies_ms.append(float((info or {}).get("inter_output_latency_ms") or 0.0))

        if is_text_stage:
            for _, info in rows:
                ttft_ms = float((info or {}).get("serving_time_to_first_output_ms") or 0.0)
                if ttft_ms > 0:
                    ttfts.append(ttft_ms / 1000.0)
                tpot_ms = float((info or {}).get("time_per_output_unit_ms") or 0.0)
                if int((info or {}).get("output_unit_count") or 0) > 1 and tpot_ms > 0:
                    tpots.append(tpot_ms / 1000.0)
                values = (info or {}).get("inter_output_latencies_ms")
                if isinstance(values, list):
                    itls.extend(float(v or 0.0) / 1000.0 for v in values)
                elif (info or {}).get("inter_output_latency_ms") is not None:
                    itls.append(float((info or {}).get("inter_output_latency_ms") or 0.0) / 1000.0)
                vllm_ttft_ms = float((info or {}).get("vllm_ttft_ms") or 0.0)
                if vllm_ttft_ms > 0:
                    vllm_ttfts.append(vllm_ttft_ms / 1000.0)
                vllm_tpot_ms = float((info or {}).get("vllm_tpot_ms") or 0.0)
                if vllm_tpot_ms > 0:
                    vllm_tpots.append(vllm_tpot_ms / 1000.0)
                vllm_values = (info or {}).get("vllm_itls_ms")
                if isinstance(vllm_values, list):
                    vllm_itls.extend(float(v or 0.0) / 1000.0 for v in vllm_values)
                elif (info or {}).get("vllm_itl_ms") is not None:
                    vllm_itls.append(float((info or {}).get("vllm_itl_ms") or 0.0) / 1000.0)
            total_output = sum(int((info or {}).get("num_tokens_out") or 0) for _, info in rows)

        audio_ttfps: list[float] = []
        audio_rtfs: list[float] = []
        audio_durations: list[float] = []
        audio_frames: list[int] = []
        missing_audio_duration_count = 0
        missing_audio_rtf_count = 0
        if is_audio_stage:
            for _, info in rows:
                audio_ttfp_ms = float((info or {}).get("serving_time_to_first_output_ms") or 0.0)
                if audio_ttfp_ms > 0:
                    audio_ttfps.append(audio_ttfp_ms / 1000.0)

                frame_count = int((info or {}).get(defs.AUDIO_FRAMES) or 0)
                if frame_count <= 0 and (info or {}).get("output_unit_type") == "audio":
                    frame_count = int((info or {}).get("output_unit_count") or 0)
                audio_frames.append(frame_count)

                duration_s = float((info or {}).get(f"{defs.AUDIO_DURATION}_s") or 0.0)
                if duration_s > 0:
                    audio_durations.append(duration_s)
                else:
                    missing_audio_duration_count += 1

                rtf = float((info or {}).get(defs.AUDIO_RTF) or 0.0)
                if rtf > 0:
                    audio_rtfs.append(rtf)
                else:
                    missing_audio_rtf_count += 1

        result.append(
            StageBenchmarkMetrics(
                stage_id=stage_id_int,
                stage_name=stage_name,
                final_output_type=final_output_type,
                total_output=total_output,
                ttfts=ttfts,
                tpots=tpots,
                itls=itls,
                vllm_ttfts=vllm_ttfts,
                vllm_tpots=vllm_tpots,
                vllm_itls=vllm_itls,
                audio_ttfps=audio_ttfps,
                audio_rtfs=audio_rtfs,
                audio_durations=audio_durations,
                audio_frames=audio_frames,
                missing_audio_duration_count=missing_audio_duration_count,
                missing_audio_rtf_count=missing_audio_rtf_count,
                stage_gen_times_ms=stage_gen_times_ms,
                postprocess_times_ms=postprocess_times_ms,
                output_unit_type=output_unit_type,
                output_unit_count=output_unit_count,
                serving_time_to_first_outputs_ms=serving_time_to_first_outputs_ms,
                time_per_output_units_ms=time_per_output_units_ms,
                inter_output_latencies_ms=inter_output_latencies_ms,
            )
        )
    return result


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: TokenizerLike | None,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
    task_type,
    selected_percentile_metrics,
    max_concurrency,
    request_rate,
    benchmark_duration,
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark.

    Args:
        input_requests: The input requests.
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        tokenizer: The tokenizer to use.
        selected_percentiles: The percentiles to select.
        goodput_config_dict: The goodput configuration.

    Returns:
        A tuple of the benchmark metrics and the actual output lengths.
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    audio_ttfps: list[float] = []
    audio_rtfs: list[float] = []
    audio_duration: list[float] = []
    audio_frames: list[int] = []
    image_generation_times_ms: list[float] = []
    denoise_step_latencies_ms: list[float] = []
    total_images = 0
    total_image_pixels = 0
    audio_underruns: list[float] = []
    audio_continuity_ok: list[bool] = []
    input_audio_duration = 0.0
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                if tokenizer is None:
                    output_len = 1
                else:
                    # We use the tokenizer to count the number of output tokens
                    # for some serving backends instead of looking at
                    # len(outputs[i].itl) since multiple output tokens may be
                    # bundled together
                    # Note : this may inflate the output token count slightly
                    output_len = len(tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += outputs[i].prompt_len
            tpot = 0
            if output_len > 1:
                try:
                    latency_minus_ttft = outputs[i].text_latency - outputs[i].ttft
                except Exception:
                    latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            audio_ttfps.append(getattr(outputs[i], defs.AUDIO_TTFP, 0.0))
            audio_rtfs.append(getattr(outputs[i], defs.AUDIO_RTF, 0.0))
            audio_duration.append(getattr(outputs[i], defs.AUDIO_DURATION, 0.0))
            audio_frames.append(getattr(outputs[i], defs.AUDIO_FRAMES, 0.0))
            image_count = int(getattr(outputs[i], "image_count", 0) or 0)
            total_images += image_count
            image_generation_time_ms = float(getattr(outputs[i], "image_generation_time_ms", 0.0) or 0.0)
            if image_generation_time_ms > 0:
                image_generation_times_ms.append(image_generation_time_ms)
            total_image_pixels += int(getattr(outputs[i], "image_pixels", 0) or 0)
            denoise_step_latency_ms = float(getattr(outputs[i], "denoise_step_latency_ms", 0.0) or 0.0)
            if denoise_step_latency_ms > 0:
                denoise_step_latencies_ms.append(denoise_step_latency_ms)
            audio_underruns.append(getattr(outputs[i], f"{defs.AUDIO_UNDERRUN}_s", 0.0))
            audio_continuity_ok.append(bool(getattr(outputs[i], defs.AUDIO_CONTINUITY_OK, True)))
            e2els.append(outputs[i].latency)
            input_audio_duration += outputs[i].input_audio_duration
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "audio_ttft" in goodput_config_dict:
            valid_metrics.append(audio_ttfps)
            slo_values.append(goodput_config_dict["audio_ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.formatwarning = lambda msg, category, filename, lineno, line=None: (
            f"{filename}:{lineno}: {category.__name__}: {msg}\n"
        )
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration on the benchmark arguments.",
            stacklevel=2,
        )

    # Calculate max output tokens per second metric
    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    # Find the time range across all successful requests
    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]
    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(output.start_time + output.latency for output in successful_outputs)

        # Create second buckets (ceiling to ensure we capture all time)
        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for i, output in enumerate(successful_outputs):
            # Calculate token generation timestamp using
            # start_time, ttft, and itl
            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)

            # Add tokens to second buckets
            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            # Track concurrent requests for each second this request was active
            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int((output.start_time + output.latency) - min_start_time)

            for second in range(request_start_second, request_end_second + 1):
                concurrent_requests_per_second[second] += 1

        # Find the maximum tokens per second and corresponding
        # concurrent requests
        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

        if TERM_PLOTLIB_AVAILABLE:
            import termplotlib as tpl

            fig = tpl.figure()
            fig.plot(
                np.arange(len(tokens_per_second)),
                tokens_per_second,
                title="Output tokens per second",
            )
            fig.plot(
                np.arange(len(concurrent_requests_per_second)),
                concurrent_requests_per_second,
                title="Concurrent requests per second",
            )
            fig.show()
        else:
            print("tip: install termplotlib and gnuplot to plot the metrics")

    metrics = MultiModalsBenchmarkMetrics(
        completed=completed,
        failed=len(failed_outputs),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by the endpoint
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles],
        mean_audio_ttfp_ms=np.mean(audio_ttfps or 0) * 1000,
        std_audio_ttfp_ms=np.std(audio_ttfps or 0) * 1000,
        median_audio_ttfp_ms=np.median(audio_ttfps or 0) * 1000,
        percentiles_audio_ttfp_ms=[(p, np.percentile(audio_ttfps or 0, p) * 1000) for p in selected_percentiles],
        mean_audio_duration_s=np.mean(audio_duration or 0),
        std_audio_duration_s=np.std(audio_duration or 0),
        median_audio_duration_s=np.median(audio_duration or 0),
        percentiles_audio_duration_s=[(p, np.percentile(audio_duration or 0, p)) for p in selected_percentiles],
        total_audio_duration_s=sum(audio_duration),
        total_audio_frames=sum(audio_frames),
        audio_throughput=sum(audio_duration) / dur_s,
        mean_audio_rtf=np.mean(audio_rtfs or 0),
        std_audio_rtf=np.std(audio_rtfs or 0),
        median_audio_rtf=np.median(audio_rtfs or 0),
        percentiles_audio_rtf=[(p, np.percentile(audio_rtfs or 0, p)) for p in selected_percentiles],
        total_images=total_images,
        image_throughput=total_images / dur_s,
        average_pixels_per_image=(total_image_pixels / total_images) if total_images > 0 else 0.0,
        mean_denoise_step_latency_ms=np.mean(denoise_step_latencies_ms or 0),
        mean_image_generation_ms=np.mean(image_generation_times_ms or 0),
        std_image_generation_ms=np.std(image_generation_times_ms or 0),
        median_image_generation_ms=np.median(image_generation_times_ms or 0),
        percentiles_image_generation_ms=[
            (p, np.percentile(image_generation_times_ms or 0, p)) for p in selected_percentiles
        ],
        mean_audio_underrun_s=np.mean(audio_underruns or 0),
        std_audio_underrun_s=np.std(audio_underruns or 0),
        median_audio_underrun_s=np.median(audio_underruns or 0),
        percentiles_audio_underrun_s=[(p, np.percentile(audio_underruns or 0, p)) for p in selected_percentiles],
        audio_continuity_ok_rate=(sum(audio_continuity_ok) / len(audio_continuity_ok)) if audio_continuity_ok else 1.0,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles],
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
        rtfx=input_audio_duration / dur_s,
    )
    print_metrics(
        task_type,
        selected_percentile_metrics,
        max_concurrency,
        request_rate,
        benchmark_duration,
        goodput_config_dict,
        metrics,
        outputs=outputs,
        selected_percentiles=selected_percentiles,
    )
    return metrics, actual_output_lens
