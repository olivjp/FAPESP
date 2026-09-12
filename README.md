# Aplicação de Algoritmos de Visão Computacional para Geração Automática de Trajetória em Selagem Robotizada Planar

**Projeto de Iniciação Científica (FAPESP / UFSCar - Departamento de Engenharia Mecânica)**
- **Candidato:** João Paulo de Oliveira Ferreira
- **Orientador:** Prof. Dr. Sidney Bruce Shiki

---

## 📁 Estrutura do Repositório

```text
FAPESP/
│
├── .vscode/                       # Configurações do ambiente de desenvolvimento no VS Code
│   ├── settings.json              # Autocomplete e inclusão automática de src no PYTHONPATH
│   └── launch.json                # Atalhos de execução e Debug (F5)
│
├── data/                          # Armazenamento organizado de dados e imagens
│   ├── raw/                       # Imagens brutas da câmera (Baumer VAX-32C.I.NVN)
│   ├── calibration/               # Imagens para calibração (tabuleiro de xadrez / ArUco)
│   ├── dataset/                   # Dataset formatado para YOLOv8 (exportado do Roboflow)
│   └── samples/                   # Imagens de exemplo para testes locais rápidos
│
├── models/                        # Modelos e pesos de Deep Learning
│   ├── pretrained/                # Pesos base (ex: yolov8n-seg.pt)
│   └── trained/                   # Modelos treinados (.pt, .onnx, TensorRT)
│
├── notebooks/                     # Jupyter Notebooks prontos para experimentação
│   ├── 01_treinamento_colab_yolov8.ipynb  # Treinamento em GPU no Google Colab + Roboflow 
│   └── 02_teste_e_validacao.ipynb         # Teste interativo das etapas no VS Code
│
├── src/                           # Módulos Python reutilizáveis do projeto
│   ├── preprocessing/             # Filtros gaussianos, equalização CLAHE e remoção de distorção
│   ├── vision/                    # Segmentação por instâncias com YOLOv8-seg
│   ├── calibration/               # Calibração intrínseca e homografia por marcadores fiduciais (ArUco)
│   ├── trajectory/                # Esqueletização, ordenamento e interpolação por B-Splines cúbicas
│   └── utils/                     # Métricas quantitativas (IoU, F1, erro de posição) e gráficos
│
├── outputs/                       # Saídas geradas pelos algoritmos
│   ├── trajectories/              # Arquivos de trajetória exportados (.csv prontos para o robô)
│   └── visual_results/            # Gráficos e imagens dos estágios do pipeline
│
├── main_pipeline.py               # Script principal ponta a ponta (Imagem -> Visão -> Trajetória 3D -> CSV)
├── requirements.txt               # Dependências para instalação via pip / venv
├── environment.yml                # Configuração para Conda / Miniconda
└── .gitignore                     # Arquivo de exclusão para Git
```

---

## 🚀 Como Configurar e Executar no VS Code

### 1. Criar e Ativar o Ambiente Virtual

#### Opção A: Usando `venv` (Padrão)
No terminal do VS Code (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Opção B: Usando Conda
```bash
conda env create -f environment.yml
conda activate fapesp-selagem
```

---

## 🏃 Como Executar o Pipeline

Para testar o pipeline completo ponta a ponta gerando uma trajetória e gráficos a partir de uma junta de selagem:

```bash
python main_pipeline.py
```

### Com argumentos personalizados:
```bash
# Executando com imagem da câmera e modelo YOLOv8 treinado:
python main_pipeline.py --image data/raw/foto_junta_01.png --model models/trained/best_sealing_yolov8n_seg.pt --scale 0.5 --z 10.0
```

---

## ☁️ Treinamento no Google Colab + Roboflow

1. Faça upload do arquivo `notebooks/01_treinamento_colab_yolov8.ipynb` para o [Google Colab](https://colab.research.google.com/).
2. Conecte em um ambiente com **GPU** (`Ambiente de Execução` -> `Alterar tipo de ambiente de execução` -> `T4 GPU`).
3. Insira sua **API Key do Roboflow** para fazer o download automático das anotações da junta de selagem.
4. Execute o treinamento. Os melhores pesos (`best.pt` e `best.onnx`) serão automaticamente salvos na sua pasta do **Google Drive**.
5. Baixe os pesos treinados e coloque-os na pasta `models/trained/` do projeto.

---

## 📊 Métricas e Validação Experimental

O projeto contempla a validação com as seguintes métricas (disponíveis em `src/utils/metrics.py`):
1. **$IoU$ (Intersection over Union)**: Qualidade da máscara de segmentação predita pela rede.
2. **$F1\text{-score}$**: Média harmônica entre precisão e revocação da detecção da junta.
3. **Erro médio de posição**: Comparação euclidiana da trajetória do efetuador em relação ao gabarito de referência (meta: $\pm 1,5$ mm).
