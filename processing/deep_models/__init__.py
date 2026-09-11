"""Prédicteurs de masques basés sur des modèles deep learning."""

from .epi import EpiMaskPredictor, process_dataset as process_epi
from .ombre import OmbreMaskPredictor, process_dataset as process_ombres
from .qpcard import QPCardMaskPredictor, process_dataset as process_qpcard
from .speculaire import SpecularMaskPredictor, process_dataset as process_speculaires
from .tige import TigeMaskPredictor, process_dataset as process_tige

__all__ = [
    "EpiMaskPredictor",
    "OmbreMaskPredictor",
    "QPCardMaskPredictor",
    "SpecularMaskPredictor",
    "TigeMaskPredictor",
    "process_epi",
    "process_ombres",
    "process_qpcard",
    "process_speculaires",
    "process_tige",
]
