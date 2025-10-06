"""categorize.py

Funções para converter bytes em imagem OpenCV e para categorizar a imagem
em um mapa de rótulos+imagem colorizada por categoria.
"""
import numpy as np
import cv2
from typing import Tuple, Dict

# cores de referência (BGR) compatíveis com OpenCV
CORES: Dict[str, Tuple[int, int, int]] = {
    "céu": (235, 206, 135),
    "pavimento": (128, 128, 128),
    "rochas": (19, 69, 139),
    "edifícios": (192, 192, 192),
    "matas": (34, 139, 34),
    "túnel": (50, 50, 50),
}

# cores de exibição em RGB
DISPLAY_COLORS = {
    "céu": (135, 206, 235),
    "pavimento": (169, 169, 169),
    "rochas": (205, 133, 63),
    "edifícios": (192, 192, 192),
    "matas": (34, 139, 34),
    "túnel": (70, 70, 70),
    "desconhecido": (255, 0, 255),
}

DEFAULT_THRESHOLD = 60.0


def bytes_para_cv2(img_bytes: bytes) -> np.ndarray:
    """Converte bytes (upload) em imagem OpenCV BGR.

    Retorna imagem em BGR (como gerado por cv2.imread).
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_cv2


def _classify_pixels_by_distance_bgr(img_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
    """Retorna label_map (H,W) com índices inteiros e mapping list para categorias.

    A classificação é feita por distância euclidiana entre BGR do pixel e CORES.
    """
    h, w = img_bgr.shape[:2]
    pixels = img_bgr.reshape(-1, 3).astype(float)
    cores_vals = np.array(list(CORES.values()), dtype=float)
    categorias = list(CORES.keys())

    # distancias vetorizadas (N, M)
    dists = np.sqrt(((pixels[:, None, :] - cores_vals[None, :, :]) ** 2).sum(axis=2))
    idx = dists.argmin(axis=1)
    dist_min = dists.min(axis=1)

    # map indices; índice M corresponde às categorias, -1 para desconhecido
    labels = np.where(dist_min <= threshold, idx, -1)
    label_map = labels.reshape(h, w)
    mapping = {i: cat for i, cat in enumerate(categorias)}
    mapping[-1] = "desconhecido"
    return label_map, mapping


def _classify_by_hsv(img_bgr: np.ndarray):
    """Classifica imagem usando regras por faixa em HSV (vetorizado).

    Retorna label_map (H,W) com índices inteiros e mapping list para categorias.
    Implementa regras heurísticas em HSV para céu, pavimento, rochas, edifícios, matas e túnel.
    """
    # converte BGR->HSV (OpenCV H:0..179 -> graus = H*2)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(int)
    S = hsv[:, :, 1].astype(int)
    V = hsv[:, :, 2].astype(int)

    H_deg = H * 2  # agora em graus 0..358

    h, w = H.shape
    label_map = -1 * np.ones((h, w), dtype=int)

    # ordem/priority: tunnel (muito escuro) -> ceu -> matas -> rochas -> pavimento -> edificios
    categorias = list(CORES.keys())

    # túnel: muito escuro
    mask_tunel = V < 50
    # céu: tons azuis claros, baixa saturação e alto valor
    mask_ceu = (H_deg >= 180 - 90) | (H_deg <= 130)  # permissivo para azuis/cianos
    mask_ceu = (H_deg >= 90) & (H_deg <= 210) & (V > 150) & (S < 90)
    # matas: verdes
    mask_matas = (H_deg >= 60) & (H_deg <= 150) & (S > 50) & (V > 40)
    # rochas: tons castanhos/bege (h pequenos ou próximos de 20-40 deg), saturação média
    mask_rochas = ((H_deg >= 0) & (H_deg <= 40) | ((H_deg >= 10) & (H_deg <= 35))) & (S > 25) & (V > 70)
    # pavimento: tons claros/grays, baixa saturação, valor médio a alto
    mask_pavimento = (S < 50) & (V >= 60) & (V < 200)
    # edificios: poupa saturação mas mais escuro que pavimento
    mask_edificios = (S < 70) & (V > 40) & (V <= 200)

    # aplicar máscaras com prioridade
    idx_map = {cat: i for i, cat in enumerate(categorias)}

    # tunnel
    if 'túnel' in idx_map:
        label_map[mask_tunel] = idx_map['túnel']

    if 'céu' in idx_map:
        label_map[(label_map == -1) & mask_ceu] = idx_map['céu']

    if 'matas' in idx_map:
        label_map[(label_map == -1) & mask_matas] = idx_map['matas']

    if 'rochas' in idx_map:
        label_map[(label_map == -1) & mask_rochas] = idx_map['rochas']

    if 'pavimento' in idx_map:
        label_map[(label_map == -1) & mask_pavimento] = idx_map['pavimento']

    if 'edifícios' in idx_map:
        label_map[(label_map == -1) & mask_edificios] = idx_map['edifícios']

    # resto continua desconhecido (-1)
    mapping = {i: cat for i, cat in enumerate(categorias)}
    mapping[-1] = 'desconhecido'
    return label_map, mapping


def label_map_to_rgb(label_map: np.ndarray, mapping: Dict[int, str]):
    """Converte label_map (H,W) para imagem RGB visual usando DISPLAY_COLORS.
    """
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, cat in mapping.items():
        color = DISPLAY_COLORS.get(cat, (255, 0, 255))
        mask = label_map == idx
        out[mask] = color
    return out


def categorize_image_from_bgr(img_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
    """Classifica imagem BGR e retorna label_map (H,W) e imagem RGB colorizada.

    Por padrão usa classificação baseada em HSV heurística. Se for desejado o método
    por distância em BGR (antigo), chame internamente _classify_pixels_by_distance_bgr.
    O parâmetro threshold aqui é usado apenas pelo método BGR-distance.
    """
    try:
        label_map, mapping = _classify_by_hsv(img_bgr)
    except Exception:
        # fallback para método por distância quando HSV falhar
        label_map, mapping = _classify_pixels_by_distance_bgr(img_bgr, threshold=threshold)

    rgb = label_map_to_rgb(label_map, mapping)
    return label_map, mapping, rgb
