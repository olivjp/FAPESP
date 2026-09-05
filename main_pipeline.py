"""
Pipeline Principal de Demonstração e Execução (Ponta a Ponta)
Projeto FAPESP / UFSCar - Selagem Robotizada Planar

Executa todas as etapas:
1. Carregamento de imagem (ou geração de junta de teste sintética)
2. Pré-processamento (Gaussiano + CLAHE)
3. Segmentação (YOLOv8-seg ou detector de borda clássico)
4. Esqueletização morfológica e ordenamento de pontos
5. Calibração e transformação para o plano cartesiano da bancada (mm)
6. Interpolação por B-spline cúbica (posições 3D, normais e tangentes)
7. Exportação da trajetória em CSV e salvamento de gráficos visuais
"""

import os
import argparse
import numpy as np
import cv2

from src.preprocessing import preprocess_frame
from src.vision import SealingJointDetector
from src.trajectory import (
    extract_skeleton_points,
    order_points_tsp,
    CubicSplineTrajectoryGenerator
)
from src.utils import (
    plot_pipeline_stages,
    plot_trajectory_2d
)


def create_synthetic_sealing_sample(filepath: str):
    """Cria uma imagem de teste sintética de junta de selagem em curva se nenhuma for fornecida."""
    img = np.full((600, 800, 3), 215, dtype=np.uint8)
    
    # Desenha uma curva em 'S' representando uma peça aeronáutica
    t = np.linspace(0, np.pi, 200)
    x = 100 + 600 * (t / np.pi)
    y = 300 + 150 * np.sin(2 * t)
    pts = np.column_stack((x, y)).astype(np.int32)

    cv2.polylines(img, [pts], isClosed=False, color=(35, 35, 35), thickness=14, lineType=cv2.LINE_AA)
    cv2.imwrite(filepath, img)
    print(f"[Amostra] Imagem sintética gerada em: {filepath}")
    return img


def run_pipeline(
    image_path: str,
    model_path: str = None,
    pixel_to_mm_scale: float = 0.5,
    z_height_mm: float = 15.0,
    output_dir: str = "outputs"
):
    print("=" * 70)
    print(" INICIANDO PIPELINE DE GERAÇÃO DE TRAJETÓRIA EM SELAGEM ROBOTIZADA")
    print("=" * 70)

    # 1. Carregamento de imagem
    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img = create_synthetic_sealing_sample(image_path)
    else:
        img = cv2.imread(image_path)
        print(f"[1. Imagem] Imagem carregada: {image_path} (Resolução: {img.shape[1]}x{img.shape[0]})")

    # 2. Pré-processamento
    print("[2. Pré-processamento] Aplicando filtro Gaussiano e normalização CLAHE...")
    prep_img = preprocess_frame(img)

    # 3. Segmentação (YOLOv8-seg)
    print("[3. Segmentação] Realizando inferência na junta de selagem...")
    detector = SealingJointDetector(model_path=model_path)
    mask, detections = detector.predict(prep_img)

    # 4. Esqueletização e Ordenamento
    print("[4. Esqueletização] Extraindo eixo central e ordenando pontos...")
    skeleton_pts = extract_skeleton_points(mask)
    if len(skeleton_pts) < 4:
        print("[Erro] Não foram detectados pontos suficientes na máscara.")
        return

    ordered_pts_px = order_points_tsp(skeleton_pts)
    print(f"[4. Esqueletização] {len(ordered_pts_px)} pontos de esqueleto ordenados com sucesso.")

    # 5. Transformação de Coordenadas (Pixels -> Milímetros no plano da bancada)
    print(f"[5. Calibração] Convertendo para mm (Fator de escala: {pixel_to_mm_scale} mm/px)...")
    points_mm = ordered_pts_px * pixel_to_mm_scale

    # Subamostragem para pontos de controle
    step = max(1, len(points_mm) // 35)
    control_points = points_mm[::step]

    # 6. Interpolação por B-Spline Cúbica
    print("[6. Trajetória] Ajustando B-spline cúbica e calculando tangentes e normais...")
    spline_gen = CubicSplineTrajectoryGenerator(smoothing_factor=4.0, default_z_mm=z_height_mm)
    trajectory_3d = spline_gen.generate_full_trajectory(control_points, num_samples=200)

    # 7. Salvamento e Exportação de Resultados
    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visual_results"), exist_ok=True)

    csv_path = os.path.join(output_dir, "trajectories", "trajetoria_selagem_gerada.csv")
    spline_gen.export_to_csv(trajectory_3d, csv_path)

    stages_plot_path = os.path.join(output_dir, "visual_results", "estagios_visao.png")
    plot_pipeline_stages(img, prep_img, mask, ordered_pts_px, save_path=stages_plot_path)

    pos_3d, tangs, norms = spline_gen.fit_spline(control_points, num_samples=200)
    traj_plot_path = os.path.join(output_dir, "visual_results", "trajetoria_cartesiana_2d.png")
    plot_trajectory_2d(control_points, pos_3d, tangs, norms, save_path=traj_plot_path)

    print("=" * 70)
    print(f"[SUCESSO] Pipeline concluído!")
    print(f" -> Trajetória exportada: {csv_path}")
    print(f" -> Gráficos salvos em: {os.path.join(output_dir, 'visual_results')}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline de visão e trajetória de selagem.")
    parser.add_argument("--image", type=str, default="data/samples/amostra_junta.png", help="Caminho da imagem de entrada.")
    parser.add_argument("--model", type=str, default=None, help="Caminho dos pesos treinados (.pt) do YOLOv8.")
    parser.add_argument("--scale", type=float, default=0.5, help="Fator de escala pixels para mm.")
    parser.add_argument("--z", type=float, default=15.0, help="Altura Z em mm da ferramenta de selagem.")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        model_path=args.model,
        pixel_to_mm_scale=args.scale,
        z_height_mm=args.z
    )
