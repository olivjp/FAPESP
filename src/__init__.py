"""
Projeto FAPESP: Aplicação de algoritmos de visão computacional para geração automática
de trajetória em selagem robotizada planar.

Módulos principais:
- preprocessing: Pré-processamento e filtragem de imagens.
- vision: Detecção e segmentação de juntas de selagem (YOLOv8-seg).
- calibration: Calibração de câmera e homografia via marcadores fiduciais.
- trajectory: Esqueletização e interpolação por B-splines cúbicas.
- utils: Métricas de avaliação e visualização gráfica.
"""

__version__ = "0.1.0"
