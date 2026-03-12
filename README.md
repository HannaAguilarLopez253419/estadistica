# ⚡ Bayesian Anomaly Analyzer

Aplicación Streamlit para análisis probabilístico bayesiano de eventos anómalos.

## 🚀 Instalación y Uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación
```bash
streamlit run app.py
```

La app abrirá en tu navegador en `http://localhost:8501`

---

## 📋 Funcionalidades

### Detección Automática de Columnas
- **Numéricas** — detectadas con `pd.api.types.is_numeric_dtype`
- **Categóricas** — columnas tipo `object` con alta cardinalidad
- **Datetime** — detección por formato y tipo `datetime64`
- **Binarias** — columnas con exactamente 2 valores únicos (0/1, Sí/No, etc.)

### Cálculo de Probabilidades
| Fórmula | Descripción |
|---------|-------------|
| `P(A)` | Probabilidad base del evento (fallo) |
| `P(B\|A)` | Verosimilitud: P(evidencia dado fallo) |
| `P(A\|B)` | Posterior via Teorema de Bayes |

### Teorema de Bayes
```
P(Fallo|Evidencia) = P(Evidencia|Fallo) × P(Fallo) / P(Evidencia)
```

### Clasificador Naive Bayes
- Entrenamiento con `GaussianNB` de sklearn
- Métricas: Accuracy, Sensibilidad, Especificidad
- Matriz de Confusión

### Visualizaciones
1. **Histogramas** de todas las variables numéricas
2. **Serie temporal** si hay columnas de fecha
3. **Gráfica de probabilidad posterior** (Prior vs Posterior)
4. **Matriz de confusión** del clasificador

---

## 📁 Dataset de Ejemplo
Se incluye `datos_ejemplo.csv` — datos de sensores industriales con:
- `timestamp` — fecha/hora de medición
- `temperatura`, `presion`, `vibracion`, `rpm`, `voltaje` — variables numéricas
- `fallo` — variable objetivo binaria (0=normal, 1=fallo)

---

## 🛠️ Estructura del Proyecto
```
bayesian_app/
├── app.py              ← Aplicación principal Streamlit
├── requirements.txt    ← Dependencias Python
├── datos_ejemplo.csv   ← Dataset de prueba
└── README.md           ← Este archivo
```
