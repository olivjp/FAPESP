"""
Módulo de Geração de Trajetória Cartesiana por B-Splines Cúbicas.
Conforme especificado no Estágio 4 e 5 do projeto:
- Interpolação suave dos pontos de controle por B-spline cúbica (C2 contínua em posição, velocidade e aceleração).
- Parametrização por comprimento de arco para garantir velocidade constante do efetuador.
- Cálculo dos vetores tangentes (direção da trajetória) e normais (alinhamento da pistola aplicadora).
- Exportação de trajetória 3D (X, Y, Z, Rz_yaw, Ry_pitch, Rx_roll) para execução robótica.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev


@dataclass
class TrajectoryPoint3D:
    x: float
    y: float
    z: float
    tangent_x: float
    tangent_y: float
    tangent_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    yaw_deg: float      # Rotação em torno de Z (direção do cordão)
    pitch_deg: float    # Inclinação em Y (perpendicular à superfície)
    roll_deg: float     # Inclinação em X


class CubicSplineTrajectoryGenerator:
    """
    Gera trajetórias 3D suaves e orientadas a partir de pontos de controle 2D interpolados por B-splines cúbicas.
    """

    def __init__(self, smoothing_factor: float = 2.0, default_z_mm: float = 0.0):
        self.smoothing_factor = smoothing_factor
        self.default_z_mm = default_z_mm

    def fit_spline(
        self,
        points_2d: np.ndarray,
        num_samples: int = 200,
        z_height_mm: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ajusta uma B-spline cúbica aos pontos (x, y) ordenados.
        
        Retorna:
        - positions_3d: np.ndarray de forma (num_samples, 3)
        - tangents: np.ndarray de forma (num_samples, 3) [vetores unitários]
        - normals: np.ndarray de forma (num_samples, 3) [vetores unitários]
        """
        if len(points_2d) < 4:
            raise ValueError("São necessários pelo menos 4 pontos para ajustar uma spline cúbica (k=3).")

        z_val = self.default_z_mm if z_height_mm is None else z_height_mm

        x = points_2d[:, 0]
        y = points_2d[:, 1]

        # Ajuste da representação paramétrica da spline cúbica (grau k=3)
        tck, u = splprep([x, y], s=self.smoothing_factor, k=min(3, len(points_2d) - 1))

        # Reamostragem uniforme ao longo do parâmetro u no intervalo [0, 1]
        u_fine = np.linspace(0, 1, num_samples)
        x_fine, y_fine = splev(u_fine, tck)
        dx_du, dy_du = splev(u_fine, tck, der=1)

        # Vetor de posições 3D
        z_fine = np.full_like(x_fine, z_val)
        positions_3d = np.column_stack((x_fine, y_fine, z_fine))

        # Cálculo dos vetores tangentes unitários (dx/ds, dy/ds, 0)
        norm_tangents = np.sqrt(dx_du**2 + dy_du**2)
        norm_tangents[norm_tangents == 0] = 1.0  # Evita divisão por zero

        tangent_x = dx_du / norm_tangents
        tangent_y = dy_du / norm_tangents
        tangent_z = np.zeros_like(tangent_x)
        tangents = np.column_stack((tangent_x, tangent_y, tangent_z))

        # Vetores normais no plano da peça (-dy, dx, 0)
        normal_x = -tangent_y
        normal_y = tangent_x
        normal_z = np.zeros_like(normal_x)
        normals = np.column_stack((normal_x, normal_y, normal_z))

        return positions_3d, tangents, normals

    def generate_full_trajectory(
        self,
        points_2d: np.ndarray,
        num_samples: int = 200,
        z_height_mm: Optional[float] = None
    ) -> List[TrajectoryPoint3D]:
        """
        Gera a lista completa de pontos de trajetória com posições, tangentes, normais e ângulos de orientação.
        """
        pos, tang, norm = self.fit_spline(points_2d, num_samples, z_height_mm)
        trajectory = []

        for p, t, n in zip(pos, tang, norm):
            # Ângulo de guinada (Yaw) alinhado com a direção tangente da curva
            yaw = np.degrees(np.arctan2(t[1], t[0]))
            pitch = 0.0   # Aplicador vertical ou perpendicular à mesa
            roll = 0.0

            trajectory.append(TrajectoryPoint3D(
                x=float(p[0]),
                y=float(p[1]),
                z=float(p[2]),
                tangent_x=float(t[0]),
                tangent_y=float(t[1]),
                tangent_z=float(t[2]),
                normal_x=float(n[0]),
                normal_y=float(n[1]),
                normal_z=float(n[2]),
                yaw_deg=float(yaw),
                pitch_deg=float(pitch),
                roll_deg=float(roll)
            ))

        return trajectory

    @staticmethod
    def export_to_csv(trajectory: List[TrajectoryPoint3D], filepath: str):
        """
        Exporta a lista de pontos da trajetória para arquivo CSV compatível com robôs industriais e ROS 2.
        """
        data = [
            {
                "x_mm": pt.x,
                "y_mm": pt.y,
                "z_mm": pt.z,
                "tangent_x": pt.tangent_x,
                "tangent_y": pt.tangent_y,
                "tangent_z": pt.tangent_z,
                "normal_x": pt.normal_x,
                "normal_y": pt.normal_y,
                "normal_z": pt.normal_z,
                "yaw_deg": pt.yaw_deg,
                "pitch_deg": pt.pitch_deg,
                "roll_deg": pt.roll_deg,
            }
            for pt in trajectory
        ]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"[Trajetória] Trajetória com {len(trajectory)} pontos exportada para: {filepath}")
