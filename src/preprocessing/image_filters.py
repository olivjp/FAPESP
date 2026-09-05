"""
Módulo de pré-processamento de imagens para o pipeline de visão computacional.
Conforme especificado no Estágio 1 do projeto:
- Remoção de ruído via filtro Gaussiano.
- Normalização de iluminação industrial (CLAHE / equalização).
- Correção de distorção de lentes (quando parâmetros intrínsecos disponíveis).
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def apply_gaussian_filter(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0) -> np.ndarray:
    """
    Aplica filtro Gaussiano para redução de ruídos de alta frequência da captura.
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)


def normalize_lighting_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para
    normalizar a intensidade luminosa em ambientes industriais com variações de luz.
    """
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    # Converte para espaço LAB para equalizar apenas o canal de luminância L
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def undistort_image(
    image: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Remove as distorções radiais e tangenciais da lente com base na matriz intrínseca.
    """
    if camera_matrix is None or dist_coeffs is None:
        return image
    return cv2.undistort(image, camera_matrix, dist_coeffs)


def preprocess_frame(
    image: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    apply_clahe: bool = True,
    gaussian_kernel: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    Executa o pipeline completo de pré-processamento de um frame RGB:
    1. Correção de distorção de lente
    2. Suavização Gaussiana
    3. Normalização de iluminação
    """
    processed = undistort_image(image, camera_matrix, dist_coeffs)
    processed = apply_gaussian_filter(processed, kernel_size=gaussian_kernel)
    if apply_clahe:
        processed = normalize_lighting_clahe(processed)
    return processed
