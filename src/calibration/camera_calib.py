"""
Módulo para calibração intrínseca da câmera (Matriz da câmera e coeficientes de distorção)
utilizando tabuleiro de xadrez (Chessboard pattern).
"""

from typing import List, Tuple, Optional
import glob
import cv2
import numpy as np


def calibrate_camera_chessboard(
    image_paths: List[str],
    board_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Calibra a câmera a partir de uma lista de imagens contendo um padrão de tabuleiro de xadrez.
    
    Retorna:
    - success: bool
    - camera_matrix: np.ndarray (3x3)
    - dist_coeffs: np.ndarray (1x5)
    - reprojection_error: float
    """
    # Prepara os pontos do objeto 3D reais no referencial do tabuleiro (Z=0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size_mm

    objpoints = []  # Pontos 3D no espaço real
    imgpoints = []  # Pontos 2D no plano da imagem
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    if len(objpoints) < 3:
        print("[Calibração Intrínseca] Pelo menos 3 imagens válidas com tabuleiro visível são necessárias.")
        return False, None, None, 0.0

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Cálculo do erro de reprojeção
    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"[Calibração Intrínseca] Sucesso! Erro médio de reprojeção: {mean_error:.4f} px")

    return True, mtx, dist, mean_error
