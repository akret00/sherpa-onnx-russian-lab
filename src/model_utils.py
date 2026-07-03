"""Модуль загрузки моделей onnx"""
import sherpa_onnx
from config import SR, pl_conf, PYANNOTE_MIN_DURATION_OFF, PYANNOTE_MIN_DURATION_ON
from vad_utils import BaseVAD, SherpaVADAdapter

def load_vad(
    vad_model: str,
    threshold: float,
    min_silence: float,
    min_speech: float,
    max_speech: float
) -> tuple[BaseVAD, int]:
    """Загружает модель VAD"""
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_model
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = min_speech
    cfg.silero_vad.max_speech_duration = max_speech
    cfg.sample_rate = SR
    cfg.provider = pl_conf.runtime.provider
    if not cfg.validate():
        raise ValueError(f"Invalid VoiceActivityDetectorConfig: {cfg}")
    # Создаем обертку над VoiceActivityDetector
    vad = SherpaVADAdapter(config = cfg, buffer_size_in_seconds = 30)
    # vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds = 30)
    window_size = cfg.silero_vad.window_size  # in samples
    return vad, window_size

def load_embedder(
    model: str,
    num_threads: int,
    provider: str = "cpu",
    debug: bool = False
) -> tuple[sherpa_onnx.SpeakerEmbeddingExtractor, sherpa_onnx.SpeakerEmbeddingManager]:
    """Загружает embedding модель"""
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model,
        num_threads=num_threads,
        debug=debug,
        provider=provider,
    )
    if not cfg.validate():
        raise ValueError(f"Invalid SpeakerEmbeddingExtractorConfig: {cfg}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    return extractor, manager

def load_asr() -> sherpa_onnx.OfflineRecognizer:
    """Загружает модель ASR"""
    if pl_conf.asr.model_type == "nemo_ctc":
        return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model = pl_conf.asr.nemo_model_path,
            tokens = pl_conf.asr.nemo_tokens_path,
            num_threads = pl_conf.runtime.num_threads,
            sample_rate = SR,
            feature_dim = 80,
            provider = pl_conf.runtime.provider,
        )

    if pl_conf.asr.model_type == "qwen3":
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend = pl_conf.asr.qwen3_conv_frontend_path,
            encoder = pl_conf.asr.qwen3_encoder_path,
            decoder = pl_conf.asr.qwen3_decoder_path,
            tokenizer = pl_conf.asr.qwen3_tokenizer_path,
            num_threads = pl_conf.runtime.num_threads,
            sample_rate = SR,
            feature_dim = 128,
            provider = pl_conf.runtime.provider,
        )

    raise ValueError(f"Unknown ASR type: {pl_conf.asr.model_type}")

def load_pyannote_diarization(
    num_speakers: int = -1,
    cluster_threshold: float = 0.6
) -> sherpa_onnx.OfflineSpeakerDiarization:
    """
    Args:
      num_speakers:
        Если известно количество спикеров в аудио, укажите их, иначе оставьте -1
      cluster_threshold:
        Если num_speakers == -1, Тогда значение этого порога используется для кластеризации.
        Маленький cluster_threshold дает больше кластеров, то есть, больше спикеров.
        Большой cluster_threshold дает меньше кластеров, то есть меньше спикеров.
    """
    pyannote_diariz_conf = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model = pl_conf.segmentation.model_path
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model = pl_conf.embed.model_path
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters = num_speakers,
            threshold = cluster_threshold
        ),
        min_duration_on = PYANNOTE_MIN_DURATION_ON,
        min_duration_off = PYANNOTE_MIN_DURATION_OFF,
    )
    if not pyannote_diariz_conf.validate():
        raise RuntimeError(
            "проверьте конфигурацию и убедитесь, что все файлы моделей присутствуют"
        )

    return sherpa_onnx.OfflineSpeakerDiarization(pyannote_diariz_conf)
