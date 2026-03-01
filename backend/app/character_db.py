from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image


class CharacterDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.registry: Dict[str, Dict[str, dict]] = {}
        self._face_detector = None
        self._embedder = None
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                self.registry = json.loads(self.db_path.read_text(encoding="utf-8"))
            except Exception:
                self.registry = {}

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(self.registry), encoding="utf-8")

    def _load_face_detector(self):
        if self._face_detector is not None:
            return self._face_detector
        try:
            from facenet_pytorch import MTCNN
            import torch

            self._face_detector = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self._face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return self._face_detector

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch

            model = InceptionResnetV1(pretrained="vggface2").eval()
            if torch.cuda.is_available():
                model = model.cuda()
            self._embedder = model
        except Exception:
            self._embedder = "hog"
        return self._embedder

    def detect_faces(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        detector = self._load_face_detector()
        out = []
        if hasattr(detector, "detect"):
            boxes, _ = detector.detect(image)
            if boxes is None:
                return out
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = max(0, x1 - 18), max(0, y1 - 18)
                x2, y2 = min(image.width, x2 + 18), min(image.height, y2 + 18)
                crop = image.crop((x1, y1, x2, y2)).resize((160, 160))
                out.append((crop, (x1, y1, x2 - x1, y2 - y1)))
            return out

        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, 1.05, 5, minSize=(28, 28))
        for (x, y, w, h) in faces:
            crop = image.crop((x, y, x + w, y + h)).resize((160, 160))
            out.append((crop, (int(x), int(y), int(w), int(h))))
        return out

    def embed_face(self, face: Image.Image) -> np.ndarray:
        embedder = self._load_embedder()
        if embedder == "hog":
            gray = cv2.cvtColor(np.array(face.convert("RGB")), cv2.COLOR_RGB2GRAY)
            feat = cv2.HOGDescriptor().compute(cv2.resize(gray, (64, 128))).flatten()
            feat = feat / (np.linalg.norm(feat) + 1e-9)
            return feat.astype(np.float32)
        import torch
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        tensor = transform(face.convert("RGB")).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            emb = embedder(tensor).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb.astype(np.float32)

    def build_series_registry(self, series_id: str, pages: List[Image.Image], min_cluster_size: int = 2) -> None:
        embeddings = []
        for page in pages:
            for face, _ in self.detect_faces(page):
                embeddings.append(self.embed_face(face))
        if not embeddings:
            logger.warning("No faces found for series {}", series_id)
            return
        arr = np.vstack(embeddings)
        labels = _cluster_embeddings(arr, min_cluster_size=min_cluster_size, eps=0.45)
        series = {}
        for label in sorted(set(labels)):
            if label == -1:
                continue
            cluster = arr[labels == label]
            centroid = cluster.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            series[f"char_{label}"] = {
                "centroid": centroid.tolist(),
                "count": int(cluster.shape[0]),
                "lora_path": "",
            }
        self.registry[series_id] = series
        self._save()
        logger.info("Series {} registered with {} characters", series_id, len(series))

    def identify(self, series_id: str, face: Image.Image, threshold: float = 0.55) -> Optional[str]:
        series = self.registry.get(series_id, {})
        if not series:
            return None
        emb = self.embed_face(face).reshape(1, -1)
        best_id = None
        best_sim = -1.0
        for char_id, data in series.items():
            centroid = np.array(data["centroid"], dtype=np.float32).reshape(1, -1)
            sim = _cosine_similarity(emb[0], centroid[0])
            if sim > best_sim:
                best_sim = sim
                best_id = char_id
        return best_id if best_sim >= threshold else None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int, eps: float) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="cosine")
        return db.fit_predict(embeddings)
    except Exception:
        return _greedy_cosine_clustering(embeddings, threshold=1.0 - eps, min_cluster_size=min_cluster_size)


def _greedy_cosine_clustering(embeddings: np.ndarray, threshold: float, min_cluster_size: int) -> np.ndarray:
    threshold = max(0.05, min(threshold, 0.95))
    labels = np.full((embeddings.shape[0],), fill_value=-1, dtype=np.int32)
    cluster_id = 0
    for idx in range(embeddings.shape[0]):
        if labels[idx] != -1:
            continue
        seed = embeddings[idx]
        members = [idx]
        for j in range(idx + 1, embeddings.shape[0]):
            if labels[j] != -1:
                continue
            if _cosine_similarity(seed, embeddings[j]) >= threshold:
                members.append(j)
        if len(members) >= min_cluster_size:
            for member in members:
                labels[member] = cluster_id
            cluster_id += 1
    return labels
