Analisador de Imagem
=====================

Fluxo principal:

1. Carregar Imagem -> Abre imagem original (BGR)
2. Analisar Cores -> Gera imagem colorizada (classificada) usando dicionário de cores
3. Exportar Imagem Colorizada -> Salva a imagem colorizada gerada pelo programa
4. Importar Imagem Categorizada -> O cliente pode editar/retocar offline e importar essa imagem
5. Definir centro / chão / topo -> Use o modo de clique (Centro/Chão/Topo) e clique na imagem ou use os sliders (funcionam em tempo real)
6. Extrapolar e Analisar -> Gera imagem extrapolada radialmente e produz CSV com contagens por setor

Dependências:
- Python 3.8+
- opencv-python
- numpy
- pandas
- matplotlib

Instalação rápida:

pip install -r requirements.txt

Como executar:

python AnalisadorImagem.py

Observações:
- Caso não importe uma imagem categorizada, a extrapolação será feita sobre a imagem colorizada gerada pelo próprio programa (depois de usar "Analisar Cores").
- O botão "Exportar Imagem Categorizada" salva a imagem extrapolada (quando presente) ou a imagem categorizada/importada; se nada disso estiver disponível, tenta salvar a imagem colorizada gerada internamente.
- Para drag & drop você pode instalar tkdnd, caso contrário use os botões para abrir arquivos.

Licença: livre para uso interno.
