# Breast Cancer XAI

Projeto desenvolvido como trabalho prático da disciplina de Redes Neurais Artificiais, da Universidade Federal de Alfenas (UNIFAL-MG). O projeto consiste na classificação de imagens histológicas de câncer de mama, utilizando Redes Neurais Convolucionais com Transfer Learning e técnicas de Explainable AI (XAI) para interpretar as decisões do modelo.

## 🎯 Objetivo

Classificar imagens histológicas de tecido mamário em benignas ou malignas, a partir do dataset **BreaKHis**, combinando diferentes níveis de magnificação da imagem e explicando as previsões do modelo por meio de técnicas de interpretabilidade.

## 🧠 Metodologia

- **Dataset**: BreaKHis (imagens histológicas de câncer de mama em múltiplas magnificações: 40X, 100X, 200X e 400X)
- **Modelos**: CNNs com Transfer Learning
  - ResNet50
  - EfficientNetB3
- **Abordagem**: Ensemble multi-magnificação, combinando as previsões dos diferentes níveis de ampliação das imagens
- **Explicabilidade (XAI)**:
  - **Grad-CAM** — geração de mapas de calor para visualizar as regiões da imagem que mais influenciaram a decisão do modelo
  - **SHAP** — atribuição de importância das features na predição

## 📂 Estrutura do projeto

```
breast_cancer/
├── src/ (ou scripts/notebooks de pré-processamento, treino, avaliação e XAI)
├── requirements.txt
├── dataset/         # ignorado pelo .gitignore (ver seção abaixo)
└── README.md
```

## ⚠️ Sobre o dataset

O dataset **BreaKHis** utilizado neste projeto **não está incluído no repositório** — a pasta correspondente está listada no `.gitignore`, pois se trata de um conjunto de imagens volumoso e de terceiros.

Para executar o projeto, é necessário baixar o dataset separadamente e organizá-lo na estrutura de pastas esperada pelo código. O BreaKHis pode ser obtido em:
🔗 https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

## 💻 Tecnologias

- Python
- TensorFlow / Keras
- Grad-CAM
- SHAP

## ▶️ Como executar

1. Clone o repositório e acesse a pasta do projeto:
   ```bash
   git clone https://github.com/isacsilveira/Redes-Neurais-Artificiais.git
   cd Redes-Neurais-Artificiais/breast_cancer
   ```
2. Baixe o dataset BreaKHis e organize-o na pasta esperada pelo projeto (não incluída no repositório).
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute os scripts/notebooks na ordem indicada (pré-processamento → treino → avaliação/XAI).

## 📄 Documentação

- **Artigo completo**: [`Artigo_Isabella.pdf`](../Artigo_Isabella.pdf) — descreve metodologia, experimentos e resultados
- **Apresentação**: [`Apresentacao_Isabella.pdf`](../Apresentacao_Isabella.pdf) — slides com resumo do projeto

## 👩‍💻 Autora

Isabella Cristina da Silveira

- 🎓 Ciência da Computação
- 🏛 Universidade Federal de Alfenas (UNIFAL-MG)
- 💻 GitHub: https://github.com/isacsilveira

Projeto desenvolvido para fins acadêmicos, como trabalho final da disciplina de Redes Neurais Artificiais da Universidade Federal de Alfenas (UNIFAL-MG).
