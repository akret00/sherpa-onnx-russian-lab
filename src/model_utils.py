"""Модуль загрузки моделей onnx"""
import sherpa_onnx
from config import config, SR

def load_vad(vad_model: str, threshold: float, min_silence: float, min_speech: float,
            max_speech: float):
    """Загружает модель VAD"""
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_model
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = min_speech
    cfg.silero_vad.max_speech_duration = max_speech
    cfg.sample_rate = SR
    cfg.provider = "cpu"
    if not cfg.validate():
        raise ValueError(f"Invalid VoiceActivityDetectorConfig: {cfg}")
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=100)
    window_size = cfg.silero_vad.window_size  # in samples
    return vad, window_size

def load_embedder(model: str, num_threads: int, provider: str = "cpu", debug: bool = False):
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

def load_asr(num_threads: int = 1, provider: str = "cpu"):
    """Загружает модель ASR"""
    if config.get_asr_model()["asr_type"] == "nemo_ctc":
        return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model = config.get_asr_model()["model"],
            tokens = config.get_asr_model()["tokens"],
            num_threads = num_threads,
            sample_rate = SR,
            feature_dim = 80,
            provider = provider,
        )

    if config.get_asr_model()["asr_type"] == "qwen3":
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend = config.get_asr_model()["conv_frontend"],
            encoder = config.get_asr_model()["encoder"],
            decoder = config.get_asr_model()["decoder"],
            tokenizer = config.get_asr_model()["tokenizer_dir"],
            num_threads = num_threads,
            sample_rate = SR,
            feature_dim = 128,
            provider = provider,
        )

    raise ValueError(f"Unknown ASR type: {config.get_asr_model()['asr_type']}")
