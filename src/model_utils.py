"""Модуль загрузки моделей onnx"""
import sherpa_onnx
from config import pl_conf, PYANNOTE_MIN_DURATION_OFF, PYANNOTE_MIN_DURATION_ON

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
