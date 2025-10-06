#!/usr/bin/env python3
"""
AnalisadorImagem.py

Aplicativo desktop (Tkinter) para análise de imagens por setores e classificação de pixels.

Funcionalidades:
- Abrir imagem (OpenCV)
- Configurar número de anéis radiais e setores angulares
- Processar imagem dividindo em setores concentricos e angulares
- Extrapolar setores que ficam fora da imagem (preenchendo com o último pixel visível)
- Mostrar imagem original e imagem classificada
- Exibir gráfico (barras) com porcentagens por categoria
- Exportar resultado consolidado para CSV
- Barra de progresso e log de status

Uso:
    python AnalisadorImagem.py

Requer:
    opencv-python, numpy, pandas, matplotlib, pillow

"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from typing import Dict, Tuple, List
import logging
import os
import threading

# módulos recém-criados
from categorize import categorize_image_from_bgr, DISPLAY_COLORS as CAT_DISPLAY_COLORS
from sectors import gerar_setores_conc_ang, desenhar_setores, metros_para_pixels, extrapolar_imagem_radial
from analyze_export import analyze_label_map_by_sectors, export_df_to_csv, classificar_por_rgb_exata_e_contar

# --- Configuração básica de logging ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# distância máxima (L2) para atribuir a cor; acima disso será 'desconhecido'
COLOR_DISTANCE_THRESHOLD = 60.0

# definição global dos ângulos dos anéis (graus) — garante consistência entre preview, extrapolação e CSV
ANGULOS_GRAUS = [2.0, 3.0, 4.0, 5.8, 8.0, 11.6, 16.6, 24.0, 36.0, 56.8]
DEFAULT_DIST_TUNEL = 90.0


def _create_classified_image(img: np.ndarray) -> np.ndarray:
    """Wrapper que usa o módulo categorize para criar imagem classificada RGB.
    Retorna imagem RGB (uint8) pronta para exibição.
    """
    # retorna rgb BGR? categorize retorna RGB visual; manter coerência exibindo RGB
    _, mapping, rgb = categorize_image_from_bgr(img, threshold=COLOR_DISTANCE_THRESHOLD)
    # categorize retorna rgb já em RGB; para manter compatibilidade com matplotlib, retornamos rgb
    return rgb


def _polar_grid_coords(h: int, w: int, max_radius: int, radial_steps: int, angular_steps: int):
    """Cria uma grade polar (r,theta) e mapeia para coordenadas image (x,y).

    A grade tem shape (radial_steps, angular_steps). r varia de 0..max_radius.
    Retorna arrays r_grid, theta_grid, x_idx, y_idx todos com mesmo shape.
    Coords x_idx/y_idx são inteiros clampados dentro do tamanho da imagem (0..w-1 / 0..h-1).
    Essa grade permite 'extrapolar' valores além do limite da imagem usando clamp.
    """
    cy, cx = h // 2, w // 2
    # radiais de 0..max_radius inclusive
    r = np.linspace(0, max_radius, radial_steps)
    theta = np.linspace(0, 2 * np.pi, angular_steps, endpoint=False)

    # grade (radial_steps, angular_steps)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")

    # mapeia para coordenadas cartesianas
    x = cx + r_grid * np.cos(theta_grid)
    y = cy + r_grid * np.sin(theta_grid)

    # arredonda e clamp
    x_idx = np.rint(x).astype(int)
    y_idx = np.rint(y).astype(int)
    x_idx = np.clip(x_idx, 0, w - 1)
    y_idx = np.clip(y_idx, 0, h - 1)

    return r_grid, theta_grid, x_idx, y_idx


def analyze_by_sectors(img: np.ndarray, radial_layers: int = 5, angular_sectors: int = 12) -> pd.DataFrame:
    """Analisa a imagem dividindo-a em setores (radial x angular) e retorna DataFrame com porcentagens.

    Usa extrapolação via grade polar com clamp (pixels além do limite usam o último pixel visível na borda).
    """
    h, w = img.shape[:2]
    max_radius = int(np.hypot(h, w) / 2)  # suficiente para cobrir cantos

    radial_steps = radial_layers
    angular_steps = angular_sectors

    # Mantemos a função para compatibilidade, mas orientamos a usar analyze_label_map_by_sectors
    raise RuntimeError("use analyze_label_map_by_sectors from analyze_export.py para análise baseada em label_map")


# -----------------------------
# Interface gráfica
# -----------------------------
class ImageAnalyzerApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Analisador de Imagem - Setores e Classificação")
        self.master.geometry("1200x770")
        self.master.minsize(1000, 600)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        # esquema de cores escuro/azul
        bg = '#071028'  # quase preto-azulado
        fg = '#ffffff'
        style.configure('.', background=bg, foreground=fg)
        self.master.configure(bg=bg)

        # estado
        self.img = None
        self.classified_img = None
        self.img_extrap = None
        self.df = None
        self.label_map = None
        self.mapping = None
        self.categorized_loaded = None  # imagem categorizada importada pelo usuário
        # opções de visualização dos setores
        self.sector_color_bgr = (0, 0, 255)  # BGR tuple usado pelo OpenCV
        self.sector_thickness = 1
        self.show_overlay = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)

        # layout: top controls, center split (left image canvases, right plots/log), bottom status
        self._build_top_controls()
        self._build_center()
        self._build_status()

    def _build_top_controls(self):
        frame = ttk.Frame(self.master)
        frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(frame, text="📂 Carregar Imagem", command=self.open_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="🎨 Analisar Cores", command=self.analyze_colors_only).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="💾 Exportar Imagem Colorizada", command=self.export_colorized_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="🚪 Sair", command=self.master.quit).pack(side=tk.RIGHT, padx=4)

        # parâmetros
        params = ttk.Frame(frame)
        params.pack(side=tk.LEFT, padx=12)

        ttk.Label(params, text="Anéis (radial):").grid(row=0, column=0, sticky=tk.E)
        self.spin_rings = ttk.Spinbox(params, from_=1, to=20, width=4)
        self.spin_rings.set(5)
        self.spin_rings.grid(row=0, column=1, padx=4)

        ttk.Label(params, text="Setores (angular):").grid(row=0, column=2, sticky=tk.E)
        self.spin_sectors = ttk.Spinbox(params, from_=4, to=72, width=4)
        self.spin_sectors.set(12)
        self.spin_sectors.grid(row=0, column=3, padx=4)

        ttk.Label(params, text="Limiar cor:").grid(row=0, column=4, sticky=tk.E)
        self.entry_threshold = ttk.Entry(params, width=5)
        self.entry_threshold.insert(0, str(int(COLOR_DISTANCE_THRESHOLD)))
        self.entry_threshold.grid(row=0, column=5, padx=4)

    # (os parâmetros de setores/câmera foram movidos para o painel direito)

    def _build_center(self):
        frame = ttk.Frame(self.master)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # esquerda: imagens (original + classificada)
        left = ttk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Painel com duas áreas: 1) Colorizar (arrastar/clique) 2) Importar imagem categorizada
        panel_top = ttk.Frame(left)
        panel_top.pack(fill=tk.X)

        self.btn_drop_raw = ttk.Button(panel_top, text='Arraste ou clique para abrir (Raw image)', command=self.open_raw_image)
        self.btn_drop_raw.pack(side=tk.LEFT, padx=6, pady=4)

        self.btn_import_categorized = ttk.Button(panel_top, text='Importar imagem categorizada', command=self.import_categorized_image)
        self.btn_import_categorized.pack(side=tk.LEFT, padx=6, pady=4)

        # tenta conectar drag & drop via tkdnd se disponível (opcional)
        try:
            import tkdnd
            dnd = tkdnd.TkDND(self.master)
            def drop_handler(event):
                data = event.data
                path = data.strip('{}')
                if path and os.path.isfile(path):
                    img = cv2.imread(path)
                    if img is not None:
                        self.img = img
                        label_map, mapping, rgb_colorized = categorize_image_from_bgr(self.img, threshold=COLOR_DISTANCE_THRESHOLD)
                        self.label_map = label_map
                        self.mapping = mapping
                        self.classified_img = rgb_colorized
                        self.ax_class.clear()
                        self.ax_class.imshow(rgb_colorized)
                        self.ax_class.set_title("Classificada", color='black')
                        self.ax_class.axis('off')
                        self.canvas_img.draw()
                        self._log(f"Arquivo arrastado e classificado: {os.path.basename(path)}")
                        self._set_status("Imagem classificada (via drag&drop)", 10)
            dnd.bindtarget(self.btn_drop_raw, drop_handler, 'text/uri-list')
        except Exception:
            pass

        # figures for images (ensure readable background)
        self.fig_img, (self.ax_orig, self.ax_class) = plt.subplots(1, 2, figsize=(8, 4))
        try:
            self.fig_img.patch.set_facecolor('white')
            self.ax_orig.set_facecolor('white')
            self.ax_class.set_facecolor('white')
        except Exception:
            pass

        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=left)
        self.canvas_img.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # connect canvas events for interactive center selection
        try:
            self.canvas_img.mpl_connect('button_press_event', self._on_canvas_click)
        except Exception:
            pass

        # set titles with explicit black color for readability on light figure
        self.ax_orig.set_title("Original", color='black')
        self.ax_orig.axis('off')
        self.ax_class.set_title("Classificada", color='black')
        self.ax_class.axis('off')

        # direita: gráfico + log
        right = ttk.Frame(frame, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # gráfico
        self.fig_plot, self.ax_plot = plt.subplots(figsize=(4, 3))
        try:
            self.fig_plot.patch.set_facecolor('white')
            self.ax_plot.set_facecolor('white')
        except Exception:
            pass
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=right)
        self.canvas_plot.get_tk_widget().pack(fill=tk.X, padx=4, pady=4)

        # log (Text) - force readable colors for log (black text on white bg)
        ttk.Label(right, text="Log / Resultados: ").pack(anchor=tk.W, padx=6)
        self.log_text = tk.Text(right, width=48, height=18, state=tk.DISABLED, fg='black', bg='white')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        # índice de cores (legend) para referência do usuário - mostra nomes e cores usadas pelo programa
        ttk.Label(right, text="Índice de cores (programa):").pack(anchor=tk.W, padx=6, pady=(6,0))
        legend_frame = ttk.Frame(right)
        legend_frame.pack(fill=tk.X, padx=6, pady=2)
        # CAT_DISPLAY_COLORS é em RGB tuples
        for cat, rgb in CAT_DISPLAY_COLORS.items():
            if cat == 'desconhecido':
                continue
            col_frame = tk.Frame(legend_frame, width=18, height=14, bg='#%02x%02x%02x' % rgb)
            col_frame.pack(side=tk.LEFT, padx=(0,4))
            lbl = ttk.Label(legend_frame, text=cat)
            lbl.pack(side=tk.LEFT, padx=(0,8))

        # controles de visualização dos setores (cor, espessura, overlay, rótulos)
        sector_ctrl = ttk.Frame(right)
        sector_ctrl.pack(fill=tk.X, padx=6, pady=(6,2))
        ttk.Label(sector_ctrl, text='Setores - visual:').grid(row=0, column=0, sticky=tk.W)
        # swatch de cor
        self._sector_swatch = tk.Label(sector_ctrl, width=3, background='#{:02x}{:02x}{:02x}'.format(self.sector_color_bgr[2], self.sector_color_bgr[1], self.sector_color_bgr[0]))
        self._sector_swatch.grid(row=0, column=1, padx=(6,8))
        ttk.Button(sector_ctrl, text='Escolher cor', command=lambda: self._choose_sector_color()).grid(row=0, column=2, sticky=tk.W)

        ttk.Label(sector_ctrl, text='Espessura:').grid(row=1, column=0, sticky=tk.W, pady=(6,0))
        self.spin_sector_thickness = ttk.Spinbox(sector_ctrl, from_=1, to=10, width=4, command=lambda: self._on_sector_thickness_changed())
        self.spin_sector_thickness.set(self.sector_thickness)
        self.spin_sector_thickness.grid(row=1, column=1, sticky=tk.W, pady=(6,0))

        self.chk_overlay = ttk.Checkbutton(sector_ctrl, text='Mostrar overlay', variable=self.show_overlay, command=self._draw_preview_overlay)
        self.chk_overlay.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(6,0))
        self.chk_labels = ttk.Checkbutton(sector_ctrl, text='Mostrar rótulos', variable=self.show_labels, command=self._draw_preview_overlay)
        self.chk_labels.grid(row=3, column=0, columnspan=2, sticky=tk.W)

        # controles específicos para a imagem categorizada (segunda etapa)
        control_categ = ttk.Frame(right)
        control_categ.pack(fill=tk.X, padx=6, pady=6)

        # click-mode selection: centro / chao / topo
        ttk.Label(control_categ, text='Clique para definir:').grid(row=0, column=0, sticky=tk.W)
        self.click_mode = tk.StringVar(value='centro')
        rb_frame = ttk.Frame(control_categ)
        rb_frame.grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(rb_frame, text='Centro', variable=self.click_mode, value='centro').pack(side=tk.LEFT)
        ttk.Radiobutton(rb_frame, text='Chão', variable=self.click_mode, value='chao').pack(side=tk.LEFT)
        ttk.Radiobutton(rb_frame, text='Topo', variable=self.click_mode, value='topo').pack(side=tk.LEFT)

        ttk.Label(control_categ, text='X Center').grid(row=1, column=0, sticky=tk.W)
        self.sld_xcenter = ttk.Scale(control_categ, from_=0, to=100, orient=tk.HORIZONTAL)
        self.sld_xcenter.grid(row=1, column=1, sticky=tk.EW, padx=4)
        self.sld_xcenter.configure(command=self._on_slider_moved)

        ttk.Label(control_categ, text='Y Chão').grid(row=2, column=0, sticky=tk.W)
        self.sld_ychao = ttk.Scale(control_categ, from_=0, to=100, orient=tk.HORIZONTAL)
        self.sld_ychao.grid(row=2, column=1, sticky=tk.EW, padx=4)
        self.sld_ychao.configure(command=self._on_slider_moved)

        ttk.Label(control_categ, text='Y Topo').grid(row=3, column=0, sticky=tk.W)
        self.sld_ytopo = ttk.Scale(control_categ, from_=0, to=100, orient=tk.HORIZONTAL)
        self.sld_ytopo.grid(row=3, column=1, sticky=tk.EW, padx=4)
        self.sld_ytopo.configure(command=self._on_slider_moved)

        ttk.Label(control_categ, text='Altura câmara (m)').grid(row=4, column=0, sticky=tk.W)
        self.entry_hcamera = ttk.Entry(control_categ, width=8)
        self.entry_hcamera.insert(0, '1.5')
        self.entry_hcamera.grid(row=4, column=1, sticky=tk.W)

        ttk.Label(control_categ, text='Altura túnel (m)').grid(row=5, column=0, sticky=tk.W)
        self.entry_htunel = ttk.Entry(control_categ, width=8)
        self.entry_htunel.insert(0, '7.0')
        self.entry_htunel.grid(row=5, column=1, sticky=tk.W)

        ttk.Label(control_categ, text='Num setores').grid(row=6, column=0, sticky=tk.W)
        self.spin_sectors_local = ttk.Spinbox(control_categ, from_=4, to=72, width=6)
        self.spin_sectors_local.set(12)
        self.spin_sectors_local.grid(row=6, column=1, sticky=tk.W)

        ttk.Button(control_categ, text='Extrapolar e Analisar', command=self.extrapolate_and_analyze).grid(row=7, column=0, columnspan=2, pady=6)

    def _build_status(self):
        frame = ttk.Frame(self.master)
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        self.status_var = tk.StringVar(value="Pronto")
        ttk.Label(frame, textvariable=self.status_var).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

    # --- utilitários da GUI ---
    def _log(self, msg: str) -> None:
        logging.info(msg)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_status(self, msg: str, progress: int = None) -> None:
        self.status_var.set(msg)
        if progress is not None:
            self.progress['value'] = progress
            self.master.update_idletasks()

    def _compute_radii_pixels(self, y_chao: int, y_topo: int, h_camera: float, h_tunel: float, dist_tunel: float = DEFAULT_DIST_TUNEL):
        """Compute a list of ring radii in pixels from ANGULOS_GRAUS and camera/tunnel params.

        Returns (lista_raios_pixels, raio_max_pixel, lista_angulos_radians)
        """
        lista_angulos = np.radians(ANGULOS_GRAUS)
        lista_raios_m = [dist_tunel * np.tan(theta) for theta in lista_angulos]
        pixels_por_metro = metros_para_pixels(y_chao, y_topo, h_camera, h_tunel)
        lista_raios_pixels = [max(1, int(r * pixels_por_metro)) for r in lista_raios_m]
        raio_max_pixel = int(max(lista_raios_pixels)) if len(lista_raios_pixels) > 0 else 1
        return lista_raios_pixels, raio_max_pixel, lista_angulos

    def _set_ui_enabled(self, enabled: bool):
        """Enable/disable primary UI controls during background processing."""
        state = tk.NORMAL if enabled else tk.DISABLED
        try:
            self.btn_drop_raw.configure(state=state)
            self.btn_import_categorized.configure(state=state)
            # top controls
            # iterate children of top frame? simpler: disable spinboxes and buttons we know
            try:
                self.spin_rings.configure(state=state)
                self.spin_sectors.configure(state=state)
                self.entry_threshold.configure(state=state)
            except Exception:
                pass
            # extrapolate button
            try:
                # control_categ Extrapolar button is a child; find by name by storing earlier would be ideal
                # fallback: disable via attribute if present
                if hasattr(self, 'spin_sectors_local'):
                    self.spin_sectors_local.configure(state=state)
            except Exception:
                pass
        except Exception:
            pass

    def _choose_sector_color(self) -> None:
        """Open a color chooser and update the sector swatch and internal BGR color."""
        # initial is RGB swatch shown as hex of BGR reversed
        try:
            initial = '#%02x%02x%02x' % (self.sector_color_bgr[2], self.sector_color_bgr[1], self.sector_color_bgr[0])
        except Exception:
            initial = '#ff0000'
        color = colorchooser.askcolor(color=initial, title='Escolha a cor dos setores')
        if color is None:
            return
        rgb_tuple, hexstr = color
        if rgb_tuple is None:
            return
        r, g, b = [int(max(0, min(255, x))) for x in rgb_tuple]
        # armazenar como BGR para OpenCV
        self.sector_color_bgr = (b, g, r)
        # atualizar swatch
        try:
            self._sector_swatch.configure(background=hexstr)
        except Exception:
            pass
        # redraw preview
        self._draw_preview_overlay()

    def _on_sector_thickness_changed(self) -> None:
        try:
            val = int(self.spin_sector_thickness.get())
            self.sector_thickness = max(1, val)
        except Exception:
            pass
        self._draw_preview_overlay()

    # --- ações ---
    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return

        self.img = img
        self.classified_img = None
        self.df = None

        # mostrar imagem original
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.ax_orig.clear()
        self.ax_orig.imshow(img_rgb)
        self.ax_orig.set_title("Original")
        self.ax_orig.axis('off')
        self.canvas_img.draw()

        self._log(f"Imagem aberta: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")
        self._set_status("Imagem carregada", 0)

    def open_raw_image(self) -> None:
        """Abre imagem raw e gera versão colorizada por categoria (primeira etapa)."""
        path = filedialog.askopenfilename(title="Selecione imagem raw", filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return

        self.img = img
        # gerar classificada (label_map,mapping,rgb)
        label_map, mapping, rgb_colorized = categorize_image_from_bgr(self.img, threshold=COLOR_DISTANCE_THRESHOLD)
        self.label_map = label_map
        self.mapping = mapping
        self.classified_img = rgb_colorized

        # exibir classificada
        self.ax_class.clear()
        self.ax_class.imshow(rgb_colorized)
        self.ax_class.set_title("Classificada")
        self.ax_class.axis('off')
        self.canvas_img.draw()

        self._log(f"Imagem raw aberta e classificada: {os.path.basename(path)}")
        self._set_status("Imagem classificada", 10)

    def import_categorized_image(self) -> None:
        """Importa imagem já categorizada/editada pelo cliente (segunda etapa)."""
        path = filedialog.askopenfilename(title="Selecione imagem categorizada", filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem categorizada.")
            return

        # assumimos imagem em BGR, mas as cores usadas para classificar devem coincidir com as cores exatas do dicionário
        self.categorized_loaded = img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax_class.clear()
        self.ax_class.imshow(img_rgb)
        self.ax_class.set_title("Importada (categorizada)")
        self.ax_class.axis('off')
        self.canvas_img.draw()

        self._log(f"Imagem categorizada importada: {os.path.basename(path)}")
        self._set_status("Imagem categorizada carregada", 10)

    def extrapolate_and_analyze(self) -> None:
        """Inicia a análise em background para não travar a GUI."""
        # prepare basic validation and defaults
        if self.categorized_loaded is None:
            if self.classified_img is None:
                messagebox.showwarning("Aviso", "Nenhuma imagem categorizada disponível. Importe uma imagem categorizada ou gere uma pela opção 'Arraste ou clique' antes de extrapolar.")
                return
            try:
                categorized_bgr = cv2.cvtColor(self.classified_img, cv2.COLOR_RGB2BGR)
            except Exception:
                categorized_bgr = self.classified_img.copy()
            self.categorized_loaded = categorized_bgr

        # read UI params synchronously (small cost)
        h, w = self.categorized_loaded.shape[:2]
        try:
            x_center_norm = float(self.sld_xcenter.get()) / 100.0
            y_chao_norm = float(self.sld_ychao.get()) / 100.0
            y_topo_norm = float(self.sld_ytopo.get()) / 100.0
            x_center = int(x_center_norm * w)
            y_chao = int(y_chao_norm * h)
            y_topo = int(y_topo_norm * h)
        except Exception:
            x_center = w // 2
            y_chao = int(h * 0.9)
            y_topo = int(h * 0.1)

        try:
            h_camera = float(self.entry_hcamera.get())
        except Exception:
            h_camera = 1.5
        try:
            h_tunel = float(self.entry_htunel.get())
        except Exception:
            h_tunel = 7.0
        try:
            num_setores = int(self.spin_sectors_local.get())
        except Exception:
            num_setores = 12

        try:
            match_thr = float(self.entry_threshold.get())
        except Exception:
            match_thr = 0.0

        # disable UI while processing
        self._set_ui_enabled(False)
        self._set_status('Executando extrapolação e contagem (em background)...', 5)

        # start background thread
        t = threading.Thread(target=self._run_extrapolate_and_analyze, args=(x_center, y_chao, y_topo, h_camera, h_tunel, num_setores, match_thr), daemon=True)
        t.start()

    def _run_extrapolate_and_analyze(self, x_center, y_chao, y_topo, h_camera, h_tunel, num_setores, match_thr):
        """Worker that performs the heavy extrapolation and counting. Posts results back to main thread."""
        try:
            # compute radii using shared ANGULOS_GRAUS
            lista_raios_pixels, raio_max_pixel, lista_angulos = self._compute_radii_pixels(y_chao, y_topo, h_camera, h_tunel)

            # extrapolate
            img_extrap = extrapolar_imagem_radial(self.categorized_loaded, (x_center, int(y_chao)), raio_max_pixel)

            # generate sector masks
            h2, w2 = img_extrap.shape[:2]
            center_extrap = (w2 // 2, h2 // 2)
            setores = gerar_setores_conc_ang(h2, w2, center_extrap, lista_raios_pixels, num_setores)

            # prepare color dict (BGR numpy arrays)
            cores_dict = {}
            for cat, rgb in CAT_DISPLAY_COLORS.items():
                if cat == 'desconhecido':
                    continue
                r, g, b = rgb
                cores_dict[cat] = np.array([b, g, r], dtype=np.uint8)

            # counting (heavy) - delegate to existing function
            df_counts = classificar_por_rgb_exata_e_contar(img_extrap, cores_dict, setores, threshold=match_thr)

            # schedule UI update on main thread
            def _finish():
                try:
                    self.df = df_counts
                    self.img_extrap = img_extrap
                    # build overlay respecting UI options
                    try:
                        if self.show_overlay.get():
                            overlay = desenhar_setores(img_extrap.copy(), center_extrap, lista_raios_pixels, num_setores, cor=tuple(self.sector_color_bgr), espessura=int(self.sector_thickness))
                        else:
                            overlay = img_extrap.copy()
                    except Exception:
                        overlay = img_extrap.copy()

                    # add labels if requested
                    if self.show_labels.get():
                        try:
                            cx, cy = center_extrap
                            for ring_idx, r_out in enumerate(lista_raios_pixels):
                                r_in = 0 if ring_idx == 0 else lista_raios_pixels[ring_idx - 1]
                                r_mid = int((r_in + r_out) / 2)
                                for s in range(num_setores):
                                    ang = 2 * np.pi * (s + 0.5) / num_setores
                                    x_text = int(cx + r_mid * np.cos(ang))
                                    y_text = int(cy + r_mid * np.sin(ang))
                                    cv2.putText(overlay, f"{ring_idx * num_setores + s + 1}", (x_text - 8, y_text + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
                                    cv2.putText(overlay, f"{ring_idx * num_setores + s + 1}", (x_text - 8, y_text + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tuple(self.sector_color_bgr), thickness=1, lineType=cv2.LINE_AA)
                        except Exception:
                            pass

                    # show overlay
                    try:
                        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    except Exception:
                        overlay_rgb = overlay
                    self.ax_orig.clear()
                    self.ax_orig.imshow(overlay_rgb)
                    self.ax_orig.set_title('Imagem Extrapolada + Setores', color='black')
                    self.ax_orig.axis('off')
                    self.canvas_img.draw()

                    # plot summary
                    cols = [c for c in df_counts.columns if c.startswith('%_')]
                    vals = [df_counts[c].mean() for c in cols]
                    cats = [c[2:] for c in cols]
                    self.ax_plot.clear()
                    bars = self.ax_plot.bar(cats, vals, color=[np.array(CAT_DISPLAY_COLORS.get(cat, (128,128,128))) / 255.0 for cat in cats])
                    try:
                        self.ax_plot.set_facecolor('white')
                        self.ax_plot.tick_params(colors='black')
                    except Exception:
                        pass
                    for bar, val in zip(bars, vals):
                        self.ax_plot.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}%", ha='center', va='bottom', color='black')
                    self.canvas_plot.draw()

                    # save automatic results
                    try:
                        import datetime
                        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        outdir = os.path.join(os.path.dirname(__file__), f'results_{ts}')
                        os.makedirs(outdir, exist_ok=True)
                        out_extrap = os.path.join(outdir, 'extrapolated.png')
                        cv2.imwrite(out_extrap, img_extrap)
                        out_overlay = os.path.join(outdir, 'overlay.png')
                        cv2.imwrite(out_overlay, overlay)
                        out_csv = os.path.join(outdir, 'counts.csv')
                        df_counts.to_csv(out_csv, index=False)
                        self._log(f'Resultados salvos em: {outdir}')
                    except Exception as e:
                        logging.exception('falha ao salvar resultados automaticamente')
                        self._log(f'Falha ao salvar resultados: {e}')

                    self._log(f'Análise via extrapolação concluída. Setores: {len(setores)}')
                    self._set_status('Pronto', 100)
                finally:
                    # re-enable UI
                    self._set_ui_enabled(True)

            # schedule finish on main thread
            self.master.after(10, _finish)

        except Exception as e:
            logging.exception('Erro na thread de extrapolação')
            def _err():
                messagebox.showerror('Erro', f'Falha na extrapolação/análise: {e}')
                self._set_ui_enabled(True)
                self._set_status('Pronto', 0)
            self.master.after(10, _err)

    def run_analysis(self) -> None:
        if self.img is None:
            messagebox.showwarning("Aviso", "Abra uma imagem primeiro.")
            return

        try:
            rings = int(self.spin_rings.get())
            sectors = int(self.spin_sectors.get())
            thr = float(self.entry_threshold.get())
        except Exception as e:
            messagebox.showerror("Erro", f"Parâmetros inválidos: {e}")
            return

        global COLOR_DISTANCE_THRESHOLD
        COLOR_DISTANCE_THRESHOLD = thr

        self._set_status("Processando...", 5)
        self._log(f"Iniciando análise: rings={rings}, sectors={sectors}, threshold={thr}")
        # passo 1: classificar imagem (gera label_map, mapping e rgb colorizada)
        self._set_status("Classificando pixels (visual)...", 10)
        try:
            label_map, mapping, rgb_colorized = categorize_image_from_bgr(self.img, threshold=COLOR_DISTANCE_THRESHOLD)
        except Exception as e:
            logging.exception("Falha na classificação")
            messagebox.showerror("Erro", f"Falha na classificação: {e}")
            return

        self.label_map = label_map
        self.mapping = mapping
        self.classified_img = rgb_colorized

        # exibir imagem classificada (rgb_colorized já é RGB pronto para matplotlib)
        self.ax_class.clear()
        self.ax_class.imshow(rgb_colorized)
        self.ax_class.set_title("Classificada", color='black')
        self.ax_class.axis('off')
        self.canvas_img.draw()

        # passo 2: gerar setores com parâmetros físicos
        self._set_status("Construindo setores...", 35)
        h, w = self.img.shape[:2]
        # parâmetros de setor e câmera - valores padrão quando campos vazios
        try:
            # usa sliders da segunda etapa, se existirem, senão usa defaults
            if hasattr(self, 'sld_xcenter'):
                x_center = int((float(self.sld_xcenter.get()) / 100.0) * w)
                y_chao = int((float(self.sld_ychao.get()) / 100.0) * h)
                y_topo = int((float(self.sld_ytopo.get()) / 100.0) * h)
            else:
                x_center = w // 2
                y_chao = int(h * 0.9)
                y_topo = int(h * 0.1)
            # alturas: pegar de entries se existirem, caso contrário usar default
            h_camera = float(self.entry_hcamera.get()) if hasattr(self, 'entry_hcamera') else 1.5
            h_tunel = float(self.entry_htunel.get()) if hasattr(self, 'entry_htunel') else 7.0
            dist_tunel = 90.0
        except Exception as e:
            messagebox.showerror("Erro", f"Parâmetros de setor inválidos: {e}")
            return

        # ângulos fixos (graus) e raios reais em metros
        angulos_graus = [2.0, 3.0, 4.0, 5.8, 8.0, 11.6, 16.6, 24.0, 36.0, 56.8]
        lista_angulos = np.radians(angulos_graus)
        lista_raios_m = [dist_tunel * np.tan(theta) for theta in lista_angulos]

        # converte metros -> pixels pela escala vertical entre chao e topo
        pixels_por_metro = metros_para_pixels(y_chao, y_topo, h_camera, h_tunel)
        lista_raios_pixels = [max(1, int(r * pixels_por_metro)) for r in lista_raios_m]

        # centro y baseado na altura da câmera
        y_center = int(y_chao - (h_camera / h_tunel) * abs(y_topo - y_chao))
        center = (x_center, y_center)

        setores_masks = gerar_setores_conc_ang(h, w, center, lista_raios_pixels, sectors)

        # desenhar setores sobre imagem (usa BGR) - fazemos cópia
        img_draw = desenhar_setores(self.img.copy(), center, lista_raios_pixels, sectors, cor=(0, 0, 255), espessura=1)
        # desenha linhas do chão e topo (verde BGR)
        cv2.line(img_draw, (0, y_chao), (w, y_chao), (0, 255, 0), 1)
        cv2.line(img_draw, (0, y_topo), (w, y_topo), (0, 255, 0), 1)

        # converter para RGB para exibir
        img_draw_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        self.ax_orig.clear()
        self.ax_orig.imshow(img_draw_rgb)
        self.ax_orig.set_title("Original + Setores", color='black')
        self.ax_orig.axis('off')
        self.canvas_img.draw()

        self._set_status("Analisando setores...", 65)
        # passo 3: analisar label_map por setores
        df = analyze_label_map_by_sectors(self.label_map, self.mapping, setores_masks)
        self.df = df

        self._set_status("Gerando gráfico...", 90)
        # plot resumo agregando porcentagens (colunas começando com %_)
        cols_pct = [c for c in df.columns if c.startswith('%_')]
        summary = df[cols_pct].sum()
        summary_percent = (summary / len(df)).to_dict()

        cats = [c[2:] for c in cols_pct]
        vals = [summary_percent.get(f"%_{cat}", 0.0) for cat in cats]

        self.ax_plot.clear()
        bars = self.ax_plot.bar(cats, vals, color=[np.array(CAT_DISPLAY_COLORS.get(cat, (128, 128, 128))) / 255.0 for cat in cats])
        self.ax_plot.set_ylabel('Percentagem média por setor (%)', color='black')
        self.ax_plot.set_title('Distribuição média por categoria', color='black')
        self.ax_plot.set_ylim(0, 100)
        try:
            self.ax_plot.tick_params(axis='x', rotation=30, colors='black')
        except Exception:
            pass
        for bar, val in zip(bars, vals):
            self.ax_plot.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%", ha='center', va='bottom', fontsize=8, color='black')
        self.canvas_plot.draw()

        # log e status final
        self._log(f"Análise concluída. Setores gerados: {len(setores_masks)}")
        self._set_status("Pronto", 100)

    def export_csv(self) -> None:
        if self.df is None or self.df.empty:
            messagebox.showwarning("Aviso", "Nenhum dado para exportar. Execute a análise primeiro.")
            return

        caminho = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')], title='Salvar resultados')
        if not caminho:
            return

        try:
            # consolidado: média por anel/sector já está em df; salvamos o DataFrame direto
            self.df.to_csv(caminho, index=False)
            messagebox.showinfo('Exportação', f'Dados salvos em:\n{caminho}')
            self._log(f'CSV salvo: {caminho}')
        except Exception as e:
            logging.exception('Falha ao salvar CSV')
            messagebox.showerror('Erro', f'Falha ao salvar CSV:\n{e}')

    def export_categorized_image(self) -> None:
        """Exporta a imagem categorizada ou extrapolada (preferência para extrapolada)."""
        # prefer extrapolated image if available
        to_save = None
        if getattr(self, 'img_extrap', None) is not None:
            to_save = self.img_extrap
        elif self.categorized_loaded is not None:
            to_save = self.categorized_loaded
        elif self.classified_img is not None:
            # classified_img is RGB (matplotlib display); convert to BGR for saving
            try:
                to_save = cv2.cvtColor(self.classified_img, cv2.COLOR_RGB2BGR)
            except Exception:
                to_save = self.classified_img.copy()

        if to_save is None:
            messagebox.showwarning('Aviso', 'Nenhuma imagem categorizada ou extrapolada disponível para salvar.')
            return

        caminho = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg;*.jpeg')], title='Salvar imagem categorizada / extrapolada')
        if not caminho:
            return

        try:
            # ensure BGR when writing with OpenCV
            save_img = to_save
            # if image looks like RGB (3rd channel order), try converting (we can't detect reliably, so attempt conversion safely)
            try:
                # a naive heuristic: if values correspond to typical matplotlib rgb (0..255) keep as BGR conversion from RGB
                save_img = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)
            except Exception:
                save_img = to_save

            cv2.imwrite(caminho, save_img)
            messagebox.showinfo('Exportação', f'Imagem salva em:\n{caminho}')
            self._log(f'Imagem salva: {caminho}')
        except Exception as e:
            logging.exception('Falha ao salvar imagem')
            messagebox.showerror('Erro', f'Falha ao salvar imagem:\n{e}')

    def analyze_colors_only(self) -> None:
        """Classifica a imagem carregada e exibe a imagem colorizada (sem criar extrapolação)."""
        if self.img is None:
            messagebox.showwarning('Aviso', 'Carregue uma imagem primeiro (Carregar Imagem).')
            return

        try:
            thr = float(self.entry_threshold.get())
        except Exception:
            thr = COLOR_DISTANCE_THRESHOLD

        self._set_status('Classificando (colorizada)...', 5)
        try:
            label_map, mapping, rgb_colorized = categorize_image_from_bgr(self.img, threshold=thr)
        except Exception as e:
            logging.exception('Falha na classificação')
            messagebox.showerror('Erro', f'Falha na classificação:\n{e}')
            return

        self.label_map = label_map
        self.mapping = mapping
        self.classified_img = rgb_colorized
        # mostrar no painel direito (ax_class)
        self.ax_class.clear()
        self.ax_class.imshow(rgb_colorized)
        self.ax_class.set_title('Classificada (colorizada)', color='black')
        self.ax_class.axis('off')
        self.canvas_img.draw()
        self._log('Imagem colorizada pronta (use Exportar Imagem Colorizada para salvar)')
        self._set_status('Pronto', 100)

    def export_colorized_image(self) -> None:
        """Exporta a imagem colorizada gerada pelo programa (self.classified_img)."""
        if self.classified_img is None:
            messagebox.showwarning('Aviso', 'Nenhuma imagem colorizada disponível. Use "Analisar Cores" primeiro.')
            return

        caminho = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg;*.jpeg')], title='Salvar imagem colorizada')
        if not caminho:
            return

        try:
            # classified_img is RGB -> convert to BGR to save
            save_img = cv2.cvtColor(self.classified_img, cv2.COLOR_RGB2BGR)
        except Exception:
            save_img = self.classified_img

        try:
            cv2.imwrite(caminho, save_img)
            messagebox.showinfo('Exportação', f'Imagem colorizada salva em:\n{caminho}')
            self._log(f'Imagem colorizada salva: {caminho}')
        except Exception as e:
            logging.exception('Falha ao salvar imagem colorizada')
            messagebox.showerror('Erro', f'Falha ao salvar imagem:\n{e}')

    def _on_slider_moved(self, _val):
        """Called when any slider moves - update preview overlay in real time."""
        self._draw_preview_overlay()

    def _on_canvas_click(self, event):
        """Handle clicks on the image canvas to set center/chao/topo depending on click_mode."""
        if event.inaxes is None:
            return
        # map click to image pixel coords
        ax = event.inaxes
        x = int(round(event.xdata))
        y = int(round(event.ydata))

        h = getattr(self.categorized_loaded, 'shape', (None,))[0]
        w = getattr(self.categorized_loaded, 'shape', (None, None))[1]
        # If categorized_loaded is None but classified_img exists we map to that
        if self.categorized_loaded is not None:
            img_w = self.categorized_loaded.shape[1]
            img_h = self.categorized_loaded.shape[0]
        elif self.classified_img is not None:
            img_w = self.classified_img.shape[1]
            img_h = self.classified_img.shape[0]
        else:
            return

        mode = self.click_mode.get()
        if mode == 'centro':
            # set slider X center
            val = int((x / img_w) * 100)
            self.sld_xcenter.set(val)
        elif mode == 'chao':
            val = int((y / img_h) * 100)
            self.sld_ychao.set(val)
        elif mode == 'topo':
            val = int((y / img_h) * 100)
            self.sld_ytopo.set(val)

        self._draw_preview_overlay()

    def _draw_preview_overlay(self):
        """Draw preview overlays (center, lines) on the currently displayed image without modifying source images."""
        # choose display image (prefer categorized_loaded, then classified_img)
        disp = None
        if self.categorized_loaded is not None:
            disp = self.categorized_loaded.copy()
        elif self.classified_img is not None:
            try:
                disp = cv2.cvtColor(self.classified_img, cv2.COLOR_RGB2BGR).copy()
            except Exception:
                disp = self.classified_img.copy()
        else:
            return

        h, w = disp.shape[:2]
        try:
            x_center = int((float(self.sld_xcenter.get()) / 100.0) * w)
            y_chao = int((float(self.sld_ychao.get()) / 100.0) * h)
            y_topo = int((float(self.sld_ytopo.get()) / 100.0) * h)
        except Exception:
            return

        # draw markers
        cv2.circle(disp, (x_center, int((y_chao + y_topo) / 2)), 6, (0, 0, 255), -1)
        cv2.line(disp, (0, y_chao), (w, y_chao), (0, 255, 0), 1)
        cv2.line(disp, (0, y_topo), (w, y_topo), (0, 255, 0), 1)

        # preview: desenhar overlay simples baseado em configurações atuais
        try:
            if self.show_overlay.get():
                # desenhar círculos concêntricos e linhas angulares com a cor selecionada
                # usar uma aproximação de 5 raios para preview
                h, w = disp.shape[:2]
                cx = int((float(self.sld_xcenter.get()) / 100.0) * w)
                cy = int((y_chao + y_topo) / 2)
                preview_radii = [int((i + 1) * min(w, h) / 10) for i in range(5)]
                try:
                    cv2.circle(disp, (cx, cy), preview_radii[0], tuple(self.sector_color_bgr), int(self.sector_thickness))
                except Exception:
                    pass
                for r in preview_radii:
                    cv2.circle(disp, (cx, cy), int(r), tuple(self.sector_color_bgr), int(self.sector_thickness))
                for j in range(12):
                    ang = 2 * np.pi * j / 12
                    x_end = int(cx + preview_radii[-1] * np.cos(ang))
                    y_end = int(cy + preview_radii[-1] * np.sin(ang))
                    cv2.line(disp, (cx, cy), (x_end, y_end), tuple(self.sector_color_bgr), int(self.sector_thickness))
                if self.show_labels.get():
                    # desenhar alguns rótulos de exemplo
                    for idx, ang in enumerate([0, 2 * np.pi / 3, 4 * np.pi / 3]):
                        x_t = int(cx + preview_radii[-1] * 0.6 * np.cos(ang))
                        y_t = int(cy + preview_radii[-1] * 0.6 * np.sin(ang))
                        cv2.putText(disp, f"{idx+1}", (x_t - 6, y_t + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
                        cv2.putText(disp, f"{idx+1}", (x_t - 6, y_t + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(self.sector_color_bgr), thickness=1, lineType=cv2.LINE_AA)
        except Exception:
            pass

        # show on ax_class
        try:
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        except Exception:
            disp_rgb = disp
        self.ax_class.clear()
        self.ax_class.imshow(disp_rgb)
        self.ax_class.set_title('Preview (posições)')
        self.ax_class.axis('off')
        self.canvas_img.draw()

def main():
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
