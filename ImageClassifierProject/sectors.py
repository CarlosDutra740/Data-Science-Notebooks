"""sectors.py

Funções para gerar setores concêntricos e angulares, desenhar sobre a imagem e
converter medidas reais para pixels (escala metros->pixels usando linha chão/topo).
"""
import numpy as np
import cv2
from typing import List, Tuple


def gerar_setores_conc_ang(h: int, w: int, center: Tuple[int, int], lista_raios_pixels: List[int], num_setores: int) -> List[np.ndarray]:
    """Gera máscaras binárias (uint8) para cada setor concêntrico e angular.

    Retorna lista de máscaras com shape (h,w) e valores 0/1.
    """
    setores = []
    Y, X = np.ogrid[:h, :w]
    dx, dy = X - center[0], Y - center[1]

    R = np.sqrt(dx ** 2 + dy ** 2)
    Theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)

    raio_in = 0
    for raio_out in lista_raios_pixels:
        for j in range(num_setores):
            ang_in = 2 * np.pi * j / num_setores
            ang_out = 2 * np.pi * (j + 1) / num_setores
            mask = (R > raio_in) & (R <= raio_out) & (Theta >= ang_in) & (Theta < ang_out)
            setores.append(mask.astype(np.uint8))
        raio_in = raio_out

    return setores


def desenhar_setores(img: np.ndarray, center: Tuple[int, int], lista_raios_pixels: List[int], num_setores: int, cor: Tuple[int, int, int] = (255, 0, 0), espessura: int = 1) -> np.ndarray:
    """Desenha círculos concêntricos e linhas angulares sobre a cópia da imagem (BGR).
    Retorna imagem BGR desenhada.
    """
    img_draw = img.copy()
    for r in lista_raios_pixels:
        cv2.circle(img_draw, center, int(r), cor, espessura)

    # divisões angulares
    for j in range(num_setores):
        ang = 2 * np.pi * j / num_setores
        x_end = int(center[0] + lista_raios_pixels[-1] * np.cos(ang))
        y_end = int(center[1] + lista_raios_pixels[-1] * np.sin(ang))
        cv2.line(img_draw, center, (x_end, y_end), cor, espessura)

    return img_draw


def metros_para_pixels(y_chao: int, y_topo: int, altura_camera_m: float, altura_tunel_m: float) -> float:
    """Calcula pixels_por_metro baseado na distância vertical entre chao e topo na imagem.

    Retorna fator pixels por metro.
    """
    if altura_tunel_m == 0:
        return 1.0
    pixels_por_metro = abs(y_topo - y_chao) / altura_tunel_m
    return pixels_por_metro


def extrapolar_imagem_radial(img: np.ndarray, center: Tuple[int, int], raio_max: int) -> np.ndarray:
    """Expande a imagem colorizada radialmente até raio_max (em pixels).

    Usa o último pixel visível em cada direção (ângulo) para preencher as áreas externas.
    Retorna imagem extrapolada com mesmo dtype e ordem de canais da entrada.
    """
    h, w = img.shape[:2]
    lado = int(2 * raio_max)
    img_extrap = np.zeros((lado, lado, 3), dtype=img.dtype)

    cx, cy = lado // 2, lado // 2

    # máximo raio presente na imagem original a partir do centro
    maxR = int(np.hypot(max(center[0], w - center[0]), max(center[1], h - center[1])))

    for y in range(lado):
        for x in range(lado):
            dx2 = x - cx
            dy2 = y - cy
            r = np.sqrt(dx2 ** 2 + dy2 ** 2)
            if r == 0:
                x_src = center[0]
                y_src = center[1]
            else:
                theta = (np.arctan2(dy2, dx2) + 2 * np.pi) % (2 * np.pi)
                r_src = min(r, maxR)
                x_src = int(center[0] + np.cos(theta) * r_src)
                y_src = int(center[1] + np.sin(theta) * r_src)

            # clamp
            x_src = np.clip(x_src, 0, w - 1)
            y_src = np.clip(y_src, 0, h - 1)

            img_extrap[y, x] = img[y_src, x_src]

    return img_extrap
