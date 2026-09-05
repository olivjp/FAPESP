"""
Módulo de Métricas e Validação Quantitativa.
Conforme especificado no Estágio 2 e Seção 4.3 (Validação) do projeto:
- Equação (1): IoU (Intersection over Union) para qualidade da máscara de segmentação.
- Equação (2): F1-score (Média harmônica entre precisão e recall).
- Erro médio de posição do efetuador / trajetória em relação à referência (Tolerância alvo: ±1.5 mm).
"""

from typing import Dict, Tuple
import numpy as np


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calcula a métrica IoU (Intersection over Union) entre duas máscaras binárias.
    IoU = Area(Interseção) / Area(União)
    """
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def calculate_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Calcula Precision, Recall e F1-score entre a máscara predita e o ground truth.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def calculate_trajectory_position_error(
    generated_points: np.ndarray,
    reference_points: np.ndarray
) -> Dict[str, float]:
    """
    Calcula o erro euclidiano ponto a ponto mais próximo entre a trajetória gerada e a referência.
    Retorna o erro médio, desvio padrão, erro máximo e a taxa de conformidade com a tolerância de ±1.5 mm.
    """
    from scipy.spatial.distance import cdist

    # Matriz de distâncias entre todos os pontos gerados e os pontos de referência
    dist_matrix = cdist(generated_points[:, :2], reference_points[:, :2])
    min_errors = np.min(dist_matrix, axis=1)

    mean_error = float(np.mean(min_errors))
    std_error = float(np.std(min_errors))
    max_error = float(np.max(min_errors))
    
    # Percentual de pontos dentro da tolerância alvo de 1.5 mm
    within_tolerance_pct = float(np.mean(min_errors <= 1.5) * 100.0)

    return {
        "mean_error_mm": mean_error,
        "std_error_mm": std_error,
        "max_error_mm": max_error,
        "within_tolerance_1_5mm_pct": within_tolerance_pct
    }
