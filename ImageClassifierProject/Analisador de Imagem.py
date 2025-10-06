import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from math import radians

# --- Cores e categorias ---
cores = {
    "céu": [135, 206, 235],
    "pavimento": [128, 128, 128],
    "rochas": [139, 69, 19],
    "edifícios": [192, 192, 192],
    "matas": [34, 139, 34],
    "túnel": [50, 50, 50],
}

# --- Classificação RGB ---
def classificar_pixel_rgb(pixel):
    for cat, cor in cores.items():
        if np.all(pixel == cor):
            return cat
    return "desconhecido"

# --- Função principal de processamento ---
def processar_imagem(img, angulos_graus):
    h, w, _ = img.shape
    centro = (w // 2, h // 2)

    # Converter para HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Criar máscara circular e setores
    resultados = []
    for ang in angulos_graus:
        raio = int(min(h, w) / 2 * np.tan(radians(ang)) / np.tan(radians(max(angulos_graus))))
        mascara = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mascara, centro, raio, 255, -1)

        pixels = img[mascara == 255]
        total = len(pixels)
        if total == 0:
            continue

        contagem = {}
        for cat, cor in cores.items():
            cor_np = np.array(cor)
            iguais = np.all(pixels == cor_np, axis=1)
            contagem[cat] = np.sum(iguais)

        resultados.append({
            "Ângulo (°)": ang,
            **{cat: contagem[cat] for cat in cores.keys()},
        })

    df = pd.DataFrame(resultados)
    return df

# --- Aplicação principal ---
class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Analisador de Imagem - Iluminação e Classificação")
        self.master.geometry("1100x700")
        self.master.configure(bg="#f4f4f4")

        style = ttk.Style()
        style.theme_use("clam")

        self.img = None
        self.df = None

        # --- Barra de controle ---
        frame_top = ttk.Frame(master)
        frame_top.pack(pady=10)

        ttk.Button(frame_top, text="📂 Abrir Imagem", command=self.abrir_imagem).grid(row=0, column=0, padx=5)
        ttk.Button(frame_top, text="⚙️ Processar", command=self.executar_processamento).grid(row=0, column=1, padx=5)
        ttk.Button(frame_top, text="💾 Exportar CSV", command=self.exportar_csv).grid(row=0, column=2, padx=5)

        # --- Canvas para imagem ---
        frame_mid = ttk.Frame(master)
        frame_mid.pack(pady=10, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_mid)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Canvas para gráfico ---
        frame_bottom = ttk.Frame(master)
        frame_bottom.pack(pady=10, fill=tk.BOTH, expand=True)
        self.fig2, self.ax2 = plt.subplots(figsize=(5,3))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=frame_bottom)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not caminho:
            return
        self.img = cv2.imread(caminho)
        if self.img is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return

        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(img_rgb)
        self.ax.set_title("Imagem Original")
        self.ax.axis("off")
        self.canvas.draw()

    def executar_processamento(self):
        if self.img is None:
            messagebox.showwarning("Aviso", "Abra uma imagem primeiro.")
            return

        angulos_graus = [2.0, 3.0, 4.0, 5.8, 8.0, 11.6, 16.6, 24.0, 36.0, 56.8]
        self.df = processar_imagem(self.img, angulos_graus)

        if self.df.empty:
            messagebox.showerror("Erro", "Nenhum dado foi processado.")
            return

        # Atualiza gráfico
        self.ax2.clear()
        categorias = list(cores.keys())
        totais = [self.df[cat].sum() for cat in categorias]
        self.ax2.bar(categorias, totais)
        self.ax2.set_title("Distribuição de Pixels por Categoria")
        self.ax2.set_ylabel("Quantidade")
        self.ax2.set_xticklabels(categorias, rotation=30)
        self.canvas2.draw()

        messagebox.showinfo("Sucesso", "Processamento concluído!")

    def exportar_csv(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("Aviso", "Nenhum dado para exportar.")
            return

        caminho = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Salvar resultados"
        )
        if caminho:
            self.df.to_csv(caminho, index=False)
            messagebox.showinfo("Exportação", f"Dados salvos em:\n{caminho}")

# --- Executar ---
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
