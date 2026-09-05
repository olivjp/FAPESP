"""
Módulo de inferência e detecção de juntas de selagem via YOLOv8-seg.
Conforme especificado no Estágio 2 e Estágio 3 do projeto:
- Inferência com modelo de segmentação por instâncias.
- Extração de máscaras binárias precisas em nível de pixel da região da junta.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import cv2


class SealingJointDetector:
    """
    Detector e segmentador de juntas de selagem baseado em YOLOv8-seg.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.4):
        """
        Inicializa o detector. Se model_path for fornecido, carrega os pesos YOLOv8.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Carrega o modelo YOLOv8 a partir do arquivo de pesos (.pt ou .onnx / engine).
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"[Detector] Modelo YOLOv8 carregado com sucesso: {model_path}")
        except ImportError:
            raise ImportError("Ultralytics não está instalado. Execute: pip install ultralytics")
        except Exception as e:
            print(f"[Detector] Erro ao carregar o modelo '{model_path}': {e}")
            self.model = None

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Realiza inferência na imagem e retorna a máscara binária combinada (0 ou 255)
        e a lista de detecções com caixas delimitadoras e confiança.
        """
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        detections = []

        if self.model is None:
            # Fallback clássico (Detecção por gradiente de Canny e morfologia)
            # útil para testes iniciais antes de treinar a rede neural
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            combined_mask = dilated
            detections.append({
                "box": [0, 0, w, h],
                "confidence": 1.0,
                "class_id": 0,
                "mode": "classic_edge_fallback"
            })
            return combined_mask, detections

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False
        )

        for result in results:
            if result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []

                for i, mask_arr in enumerate(masks_data):
                    # Redimensiona a máscara para a resolução original da imagem
                    mask_resized = cv2.resize(
                        (mask_arr * 255).astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    combined_mask = np.bitwise_or(combined_mask, mask_resized)

                    det_info = {
                        "box": boxes[i].tolist() if i < len(boxes) else None,
                        "confidence": float(confs[i]) if i < len(confs) else 1.0,
                        "class_id": int(classes[i]) if i < len(classes) else 0,
                        "mode": "yolov8_seg"
                    }
                    detections.append(det_info)

        return combined_mask, detections
