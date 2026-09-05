"""
Módulo de Visualização Gráfica do Pipeline e Trajetórias.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_pipeline_stages(
    original_img: np.ndarray,
    preprocessed_img: np.ndarray,
    mask: np.ndarray,
    skeleton_pts_px: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plota as 4 etapas de visão: Original -> Pré-processado -> Máscara de Segmentação -> Esqueleto.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) if len(original_img.shape) == 3 else original_img
    prep_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB) if len(preprocessed_img.shape) == 3 else preprocessed_img

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Imagem Original")
    axes[0].axis("off")

    axes[1].imshow(prep_rgb)
    axes[1].set_title("2. Pré-processada (CLAHE + Gaussiano)")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("3. Máscara YOLOv8-seg")
    axes[2].axis("off")

    axes[3].imshow(mask, cmap="gray")
    if len(skeleton_pts_px) > 0:
        axes[3].scatter(skeleton_pts_px[:, 0], skeleton_pts_px[:, 1], c="red", s=2, label="Esqueleto")
    axes[3].set_title("4. Eixo Central (Esqueletização)")
    axes[3].legend(loc="upper right")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualizador] Gráfico salvo em: {save_path}")
    plt.close()


def plot_trajectory_2d(
    control_points_mm: np.ndarray,
    spline_positions_mm: np.ndarray,
    tangents: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    quiver_step: int = 15,
    save_path: Optional[str] = None
):
    """
    Plota a trajetória interpolada por B-spline no plano real da mesa (em mm) com vetores tangentes e normais.
    """
    plt.figure(figsize=(9, 7))

    plt.scatter(
        control_points_mm[:, 0],
        control_points_mm[:, 1],
        c="gray",
        alpha=0.6,
        s=15,
        label="Pontos de Controle (Esqueleto)"
    )

    plt.plot(
        spline_positions_mm[:, 0],
        spline_positions_mm[:, 1],
        "b-",
        linewidth=2.5,
        label="Trajetória Suave (B-spline Cúbica)"
    )

    # Plota vetores de orientação (tangente e normal)
    if tangents is not None and normals is not None:
        sub_pos = spline_positions_mm[::quiver_step]
        sub_tang = tangents[::quiver_step]
        sub_norm = normals[::quiver_step]

        plt.quiver(
            sub_pos[:, 0], sub_pos[:, 1],
            sub_tang[:, 0], sub_tang[:, 1],
            color="red", scale=25, width=0.004, label="Vetor Tangente (Velocidade)"
        )
        plt.quiver(
            sub_pos[:, 0], sub_pos[:, 1],
            sub_norm[:, 0], sub_norm[:, 1],
            color="green", scale=25, width=0.004, label="Vetor Normal (Alinhamento)"
        )

    plt.title("Trajetória Cartesiana Gerada para Selagem (Plano da Mesa)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualizador] Trajetória 2D salva em: {save_path}")
    plt.close()


def plot_trajectory_comparison(
    generated_pts: np.ndarray,
    reference_pts: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Compara a trajetória gerada com uma trajetória de referência (Ground Truth) para análise de erro.
    """
    plt.figure(figsize=(9, 7))
    plt.plot(reference_pts[:, 0], reference_pts[:, 1], "k--", linewidth=2, label="Referência Teórica (Gabarito)")
    plt.plot(generated_pts[:, 0], generated_pts[:, 1], "r-", linewidth=2, label="Trajetória Gerada por Visão")

    plt.title("Validação de Trajetória: Gerada vs Referência")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualizador] Comparação salva em: {save_path}")
    plt.close()
