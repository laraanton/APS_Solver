# Online Shoppers Purchase Intention — APS Solver

Proyecto de machine learning para predecir si un usuario realizará una compra en una tienda online, basándose en su comportamiento de navegación.

---

## Contexto del problema

Una tienda online internacional detecta que muchos usuarios navegan y añaden productos al carrito pero no finalizan la compra. El objetivo es construir un modelo predictivo que identifique la **intención de compra** (`Revenue = True/False`) a partir de variables de sesión y comportamiento, para que el equipo de marketing pueda actuar en consecuencia.

El dataset proviene del [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) e incluye ~12.000 sesiones de usuarios con variables como duración de visita, tipo de página, mes, tipo de visitante y más.

---

## Estructura del repositorio

```
├── data/
│   └── online_shoppers.csv         # Dataset de entrada (para estudiantes)
├── models/
│   ├── random_forest.py            # Clasificador Random Forest
│   ├── knn.py                      # K-Nearest Neighbors
│   ├── mlp.py                      # Red neuronal (MLP)
│   ├── perceptron.py               # Perceptrón simple
│   └── logistic_regression.py      # Regresión logística
├── notebooks/
│   └── EDA.ipynb                   # Análisis exploratorio de datos
├── solver.py                       # Clase principal APS_Solver (RF + clustering)
└── README.md
```

---

## Instalación

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Uso rápido

```python
from solver import APS_Solver
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar y dividir datos
df = pd.read_csv("data/online_shoppers.csv")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Revenue"])
train_df.to_csv("online_shoppers_train.csv", index=False)
test_df.to_csv("online_shoppers_test.csv", index=False)

# Entrenar y evaluar
model = APS_Solver()
model.train_model("online_shoppers_train.csv")
model.test_model("online_shoppers_test.csv")
```

---

## Modelos disponibles

Cada archivo en `models/` implementa la misma interfaz (`train_model`, `test_model`, `cluster_data`) con un clasificador distinto:

| Archivo | Algoritmo |
|---|---|
| `random_forest.py` | Random Forest |
| `knn.py` | K-Nearest Neighbors |
| `mlp.py` | Red neuronal multicapa |
| `perceptron.py` | Perceptrón (Rosenblatt) |
| `logistic_regression.py` | Regresión logística |

---

## Pipeline de datos

1. **Preprocesado** — eliminación de duplicados, imputación de nulos, normalización de categorías con `difflib`, One-Hot Encoding y escalado `StandardScaler`.
2. **Clustering** — KMeans sobre el dataset completo con visualización PCA 2D para análisis de segmentos.
3. **Clasificación** — entrenamiento con split 70/30 estratificado y evaluación con precisión, recall y F1-score.

---

## Métricas de evaluación

Todos los modelos reportan:
- Tasa de error
- Precisión
- Recall
- F1-score
