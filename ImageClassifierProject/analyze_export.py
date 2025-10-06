"""analyze_export.py

Funções para analisar uma imagem categorizada (label_map) usando máscaras de setores
e exportar os resultados para CSV.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


def analyze_label_map_by_sectors(label_map: np.ndarray, mapping: Dict[int, str], setores: List[np.ndarray]) -> pd.DataFrame:
    """Analisa um label_map (H,W) para cada máscara de setor e retorna DataFrame.

    Cada setor é uma máscara (H,W) com valores 0/1. mapping: índice -> categoria string.
    O DataFrame contém colunas: sector_id, total_pixels, count_<categoria>, %_<categoria>
    """
    records = []
    categorias = sorted(set(mapping.values()))

    for i, mask in enumerate(setores, start=1):
        mask_bool = mask.astype(bool)
        total = int(mask_bool.sum())
        counts = {}
        for idx, cat in mapping.items():
            counts[cat] = int((label_map == idx)[mask_bool].sum())

        pct = {f"%_{cat}": (counts[cat] / total * 100) if total > 0 else 0.0 for cat in categorias}

        rec = {
            "sector_id": i,
            "total_pixels": total,
            **{f"count_{cat}": counts[cat] for cat in categorias},
            **pct,
        }
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df


def export_df_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def classificar_por_rgb_exata_e_contar(img_colorizada: np.ndarray, cores: Dict[str, np.ndarray], setores: List[np.ndarray], threshold: float = 0.0) -> pd.DataFrame:
    """Classifica imagem colorizada por correspondência exata ou tolerante e conta por setor.

    - img_colorizada: imagem BGR (como lida por cv2.imread / cv2.imwrite)
    - cores: dict cat -> np.array([B,G,R])
    - setores: lista de máscaras (H,W)
    - threshold: se 0.0 faz comparação exata; se >0.0 usa distância L2 e aceita correspondências <= threshold.

    Retorna DataFrame com colunas: sector_id, total_pixels, %_<categoria> ...
    """
    h, w = img_colorizada.shape[:2]
    records = []
    categorias = list(cores.keys())

    # converte cores de referência em array (M,3)
    cores_vals = np.array([cores[cat] for cat in categorias], dtype=float)

    for i, mask in enumerate(setores, start=1):
        mask_bool = mask.astype(bool)
        pixels = img_colorizada[mask_bool].reshape(-1, 3).astype(float)
        total = pixels.shape[0]
        pcts = {cat: 0.0 for cat in categorias}

        if total > 0:
            if threshold <= 0.0:
                # comparação exata
                for idx, cat in enumerate(categorias):
                    cor = cores_vals[idx].astype(np.uint8)
                    iguais = np.all(pixels.astype(np.uint8) == cor, axis=1)
                    pcts[cat] = float(np.sum(iguais) / total * 100)
            else:
                # distância L2 para cada pixel vs cada cor -> (N, M)
                dists = np.sqrt(((pixels[:, None, :] - cores_vals[None, :, :]) ** 2).sum(axis=2))
                idx_min = dists.argmin(axis=1)
                dist_min = dists.min(axis=1)
                for idx, cat in enumerate(categorias):
                    match = (idx_min == idx) & (dist_min <= threshold)
                    pcts[cat] = float(np.sum(match) / total * 100)

        rec = {"sector_id": i, "total_pixels": int(total), **{f"%_{cat}": pcts[cat] for cat in categorias}}
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df
