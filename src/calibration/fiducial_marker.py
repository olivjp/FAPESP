"""
Módulo de Calibração com Marcadores Fiduciais (ArUco / Homografia Planar).
Conforme Estágio 3 do projeto:
- Conversão do referencial de imagem (pixels) para o referencial de bancada/robô (mm).
- Fixação da origem (0, 0, z0) via marcador fiducial ou pontos de controle conhecidos.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class ArUcoCalibrator:
    """
    Detecta marcadores ArUco e calcula a transformação de homografia 2D (pixels -> mm).
    """

    def __init__(
        self,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
        marker_length_mm: float = 50.0
    ):
        self.dictionary_id = dictionary_id
        self.marker_length_mm = marker_length_mm
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.homography_matrix: Optional[np.ndarray] = None

    def detect_markers(self, image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Detecta os marcadores ArUco na imagem fornecida.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def compute_homography_from_aruco(
        self,
        image: np.ndarray,
        target_id: int = 0,
        origin_x_mm: float = 0.0,
        origin_y_mm: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Calcula a homografia baseada nos 4 cantos de um marcador ArUco de tamanho conhecido.
        Define o canto superior esquerdo do marcador como (origin_x_mm, origin_y_mm).
        """
        corners, ids = self.detect_markers(image)
        if ids is None or len(ids) == 0:
            print("[Calibração] Nenhum marcador ArUco detectado.")
            return None

        found_idx = None
        for i, m_id in enumerate(ids.flatten()):
            if m_id == target_id:
                found_idx = i
                break

        if found_idx is None:
            print(f"[Calibração] Marcador ID {target_id} não encontrado na imagem.")
            return None

        img_pts = corners[found_idx][0].astype(np.float32)  # 4 pontos: top-left, top-right, bottom-right, bottom-left
        
        l = self.marker_length_mm
        obj_pts = np.array([
            [origin_x_mm, origin_y_mm],
            [origin_x_mm + l, origin_y_mm],
            [origin_x_mm + l, origin_y_mm + l],
            [origin_x_mm, origin_y_mm + l]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(img_pts, obj_pts)
        self.homography_matrix = H
        return H

    def set_manual_homography(self, img_points: np.ndarray, world_points_mm: np.ndarray) -> np.ndarray:
        """
        Calcula a homografia a partir de 4 ou mais pares de pontos conhecidos (Pixels <-> mm).
        """
        H, _ = cv2.findHomography(img_points.astype(np.float32), world_points_mm.astype(np.float32))
        self.homography_matrix = H
        return H


def transform_points_homography(points_2d_px: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    """
    Aplica a matriz de homografia em um conjunto de pontos 2D (N, 2) em pixels
    para obter as coordenadas no plano real em milímetros (N, 2).
    """
    if points_2d_px.ndim == 1:
        points_2d_px = points_2d_px.reshape(-1, 2)

    num_pts = len(points_2d_px)
    pts_hom = np.hstack([points_2d_px, np.ones((num_pts, 1), dtype=np.float32)])
    
    transformed = (homography_matrix @ pts_hom.T).T
    transformed[:, 0] /= transformed[:, 2]
    transformed[:, 1] /= transformed[:, 2]
    
    return transformed[:, :2]
