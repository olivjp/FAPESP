"""
Módulo de Esqueletização e Ordenamento de Pontos 2D.
Conforme especificado no Estágio 3 do projeto:
- Redução da máscara binária ao seu eixo central com espessura de 1 pixel.
- Preservação da topologia e geometria da junta de selagem.
- Ordenamento sequencial dos pontos em uma cadeia contínua de trajetória.
"""

from typing import List, Tuple
import cv2
import numpy as np


def extract_skeleton_points(mask: np.ndarray) -> np.ndarray:
    """
    Extrai o esqueleto morfológico (eixo central de 1 pixel) da máscara binária.
    Retorna os pontos (x, y) em coordenadas de pixel.
    """
    # Garante imagem binária 0 e 255
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Aplica esqueletização morfológica de Zhang-Suen ou Thinning do OpenCV
    try:
        skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        # Fallback usando scikit-image ou algoritmo morfológico iterativo do OpenCV
        try:
            from skimage.morphology import skeletonize
            skeleton = (skeletonize(binary > 0) * 255).astype(np.uint8)
        except ImportError:
            # Implementação manual pura com OpenCV
            skeleton = np.zeros(binary.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp_bin = binary.copy()
            while True:
                eroded = cv2.erode(temp_bin, element)
                opened = cv2.dilate(eroded, element)
                temp = cv2.subtract(temp_bin, opened)
                skeleton = cv2.bitwise_or(skeleton, temp)
                temp_bin = eroded.copy()
                if cv2.countNonZero(temp_bin) == 0:
                    break

    # Extrai coordenadas (x, y) dos pixels brancos
    y_indices, x_indices = np.where(skeleton > 0)
    if len(x_indices) == 0:
        return np.empty((0, 2), dtype=np.float32)

    points = np.column_stack((x_indices, y_indices)).astype(np.float32)
    return points


def order_points_tsp(points: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """
    Ordena um conjunto de pontos 2D não ordenados usando o algoritmo do vizinho mais próximo
    (Nearest Neighbor) para formar uma trajetória sequencial contínua.
    """
    if len(points) <= 1:
        return points

    num_points = len(points)
    visited = np.zeros(num_points, dtype=bool)
    ordered_indices = [start_idx]
    visited[start_idx] = True

    current_idx = start_idx
    for _ in range(num_points - 1):
        dists = np.linalg.norm(points - points[current_idx], axis=1)
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        ordered_indices.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx

    return points[ordered_indices]
