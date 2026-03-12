import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APP Probabilidades",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:      #fdf6fb;
    --surface: #f9eef6;
    --card:    #ffffff;
    --border:  #e8d5f0;
    --accent:  #c084fc;
    --accent2: #67e8f9;
    --accent3: #fbbf24;
    --accent4: #f9a8d4;
    --text:    #3b1f4e;
    --muted:   #9d7bb0;
    --danger:  #fb7185;
    --success: #6ee7b7;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3, h4 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    color: var(--text) !important;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(192,132,252,0.10);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent4));
    border-radius: 16px 16px 0 0;
}

.metric-value {
    font-family: 'DM Mono', monospace !important;
    font-size: 2rem;
    font-weight: 500;
    color: #a855f7;
    line-height: 1;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-top: 6px;
}

.insight-box {
    background: linear-gradient(135deg, #fdf2ff 0%, #fce7f3 100%);
    border: 1px solid #e9d5ff;
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    font-family: 'Nunito', sans-serif;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text);
}

.insight-box .insight-header {
    color: #9333ea;
    font-weight: 800;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 8px;
}

.prob-bar-container {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin: 6px 0;
}

.prob-bar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 6px;
}

.prob-bar-track {
    background: #f3e8ff;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent4));
    transition: width 0.5s ease;
}

.prob-value {
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    font-size: 1.1rem;
    color: #a855f7;
    float: right;
    margin-top: -22px;
}

.section-header {
    border-left: 4px solid var(--accent);
    padding-left: 12px;
    margin: 30px 0 16px 0;
}

.tag {
    display: inline-block;
    background: #fdf4ff;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    margin: 2px;
}

.tag.numeric    { border-color: #67e8f9; color: #0891b2; background: #ecfeff; }
.tag.categorical{ border-color: #fbbf24; color: #b45309; background: #fffbeb; }
.tag.datetime   { border-color: #6ee7b7; color: #059669; background: #ecfdf5; }
.tag.binary     { border-color: #fda4af; color: #e11d48; background: #fff1f2; }

[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #c084fc, #f0abfc) !important;
    color: #3b1f4e !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0.03em !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(192,132,252,0.35) !important;
}

.stFileUploader {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
}

div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.title-hero {
    text-align: center;
    padding: 36px 0 16px 0;
}

.title-hero h1 {
    font-size: 3.2rem !important;
    background: linear-gradient(135deg, #c084fc, #f472b6, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1 !important;
    margin-bottom: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
}

.title-hero p {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 24px 0;
}

</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB THEME ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#fdf6fb',
    'axes.facecolor':   '#ffffff',
    'axes.edgecolor':   '#e8d5f0',
    'axes.labelcolor':  '#9d7bb0',
    'text.color':       '#3b1f4e',
    'xtick.color':      '#9d7bb0',
    'ytick.color':      '#9d7bb0',
    'grid.color':       '#f3e8ff',
    'grid.alpha':       0.8,
    'font.family':      'sans-serif',
    'axes.titlecolor':  '#3b1f4e',
    'axes.titlesize':   11,
    'axes.labelsize':   9,
})

PALETTE = ['#c084fc', '#f9a8d4', '#67e8f9', '#6ee7b7', '#fbbf24', '#fb7185', '#a78bfa']

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

@st.cache_data
def load_csv(file):
    try:
        # Intentar diferentes separadores comunes
        content = file.read()
        file.seek(0)

        if len(content.strip()) == 0:
            st.error("❌ El archivo está vacío. Por favor sube un CSV con datos.")
            st.stop()

        # Detectar separador automáticamente
        sample = content[:2000].decode('utf-8', errors='replace')
        sep = ','
        if sample.count(';') > sample.count(','):
            sep = ';'
        elif sample.count('\t') > sample.count(','):
            sep = '\t'

        file.seek(0)
        df = pd.read_csv(file, sep=sep, encoding='utf-8', on_bad_lines='skip')

        if df.empty or len(df.columns) == 0:
            st.error("❌ No se pudieron leer columnas del archivo. Verifica que sea un CSV válido.")
            st.stop()

        return df

    except UnicodeDecodeError:
        file.seek(0)
        try:
            df = pd.read_csv(file, sep=sep, encoding='latin-1', on_bad_lines='skip')
            if df.empty:
                st.error("❌ Archivo vacío o sin columnas válidas.")
                st.stop()
            return df
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error inesperado al cargar el archivo: {e}")
        st.stop()

def detect_columns(df):
    numeric_cols, categorical_cols, datetime_cols, binary_cols = [], [], [], []
    for col in df.columns:
        # datetime
        if df[col].dtype == 'datetime64[ns]':
            datetime_cols.append(col)
            continue
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            if df[col].dtype == object:
                sample = df[col].dropna().head(20)
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:
                    datetime_cols.append(col)
                    continue
        except:
            pass

        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)

        # binary
        if n_unique <= 2:
            str_vals = set(str(v).strip().lower() for v in unique_vals)
            binary_indicators = [{'0','1'}, {'true','false'}, {'yes','no'}, {'sí','no'}, {'si','no'}, {'1.0','0.0'}]
            if any(str_vals <= bi for bi in binary_indicators) or n_unique <= 2:
                binary_cols.append(col)
                continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols, binary_cols

def compute_probabilities(df, target_col, evidence_col=None, threshold=None):
    target = df[target_col].dropna()
    # ensure binary 0/1
    unique_t = sorted(target.unique())
    if len(unique_t) != 2:
        return None
    val_positive = unique_t[1]
    target_binary = (target == val_positive).astype(int)

    n = len(target_binary)
    p_a = target_binary.mean()  # P(Fallo)
    results = {'P(A)': p_a, 'P(no_A)': 1 - p_a}

    if evidence_col and threshold is not None:
        ev_series = df[evidence_col].dropna()
        aligned = target_binary.reindex(ev_series.index).dropna()
        ev_aligned = ev_series.reindex(aligned.index)

        b = (ev_aligned > threshold).astype(int)
        p_b = b.mean()  # P(B)

        both = ((aligned == 1) & (b == 1))
        p_ab = both.mean()  # P(A∩B)

        p_b_given_a = both.sum() / aligned.sum() if aligned.sum() > 0 else 0  # P(B|A)
        p_a_given_b = (p_b_given_a * p_a / p_b) if p_b > 0 else 0             # Bayes: P(A|B)

        results.update({
            'P(B)': p_b,
            'P(A∩B)': p_ab,
            'P(B|A)': p_b_given_a,
            'P(A|B)_Bayes': p_a_given_b,
            'threshold': threshold,
            'evidence_col': evidence_col,
            'target_positive': val_positive
        })

    return results

def run_naive_bayes(df, target_col, feature_cols):
    df_model = df[feature_cols + [target_col]].dropna()
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

    # encode if needed
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y.astype(str))

    if len(np.unique(y_enc)) < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, cm[1,1])
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'cm': cm, 'acc': acc, 'sensitivity': sens, 'specificity': spec,
        'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
        'classes': le_y.classes_
    }

def generate_insights(df, target_col, probs, nb_results):
    insights = []

    p_a = probs.get('P(A)', 0)
    if p_a < 0.05:
        insights.append(f"🔵 Evento RARO: La tasa de '{target_col}' es solo {p_a:.1%}. Considera técnicas de balanceo de clases.")
    elif p_a > 0.4:
        insights.append(f"🔴 Alta frecuencia: '{target_col}' ocurre en {p_a:.1%} de los casos. El sistema puede estar en estado crítico.")
    else:
        insights.append(f"🟡 Frecuencia moderada: '{target_col}' ocurre en {p_a:.1%} de los registros.")

    if 'P(A|B)_Bayes' in probs and 'P(B|A)' in probs:
        lift = probs['P(A|B)_Bayes'] / p_a if p_a > 0 else 1
        insights.append(f"📈 Lift bayesiano: {lift:.2f}x — la evidencia '{probs['evidence_col']} > {probs['threshold']:.2f}' "
                        f"{'AUMENTA' if lift > 1 else 'REDUCE'} la probabilidad de fallo en {abs(lift-1):.0%}.")
        insights.append(f"⚡ P(Fallo|Evidencia) = {probs['P(A|B)_Bayes']:.4f} vs P(Fallo) base = {p_a:.4f}")

    if nb_results:
        acc = nb_results['acc']
        sens = nb_results['sensitivity']
        spec = nb_results['specificity']
        if acc > 0.85:
            insights.append(f"✅ Clasificador Bayesiano: Accuracy {acc:.1%} — Modelo CONFIABLE para predicción.")
        elif acc > 0.65:
            insights.append(f"⚠️ Clasificador Bayesiano: Accuracy {acc:.1%} — Rendimiento ACEPTABLE, considerar más features.")
        else:
            insights.append(f"❌ Clasificador Bayesiano: Accuracy {acc:.1%} — Rendimiento BAJO. Datos pueden ser insuficientes.")
        insights.append(f"🎯 Sensibilidad: {sens:.1%} | Especificidad: {spec:.1%} — "
                        f"{'Alta detección de fallos reales' if sens > 0.7 else 'Muchos fallos reales no detectados'}.")

    return insights

# ─── PLOT FUNCTIONS ──────────────────────────────────────────────────────────────

def plot_histograms(df, numeric_cols, target_col):
    cols_to_plot = [c for c in numeric_cols if c != target_col][:6]
    if not cols_to_plot:
        return None
    n = len(cols_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, color=PALETTE[i % len(PALETTE)], alpha=0.8, edgecolor='none')
        ax.axvline(data.mean(), color='#f59e0b', linewidth=1.5, linestyle='--', label=f'μ={data.mean():.2f}')
        ax.set_title(col, pad=8)
        ax.set_xlabel('')
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribución de Variables Numéricas', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig

def plot_temporal(df, datetime_cols, target_col):
    if not datetime_cols:
        return None
    date_col = datetime_cols[0]
    try:
        df_t = df.copy()
        df_t[date_col] = pd.to_datetime(df_t[date_col], errors='coerce')
        df_t = df_t.dropna(subset=[date_col]).sort_values(date_col)

        fig, ax = plt.subplots(figsize=(12, 4))
        target_vals = pd.to_numeric(df_t[target_col], errors='coerce').fillna(0)
        ax.fill_between(df_t[date_col], target_vals, alpha=0.3, color='#7c3aed')
        ax.plot(df_t[date_col], target_vals, color='#7c3aed', linewidth=1.5)

        # Mark anomalies
        anomalies = df_t[target_vals == 1]
        if len(anomalies) > 0:
            ax.scatter(anomalies[date_col], [1]*len(anomalies), color='#ef4444', zorder=5,
                      s=40, marker='v', label='Evento detectado')

        ax.set_title(f'Serie Temporal — {target_col}', pad=10)
        ax.set_xlabel(date_col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except:
        return None

def plot_posterior_comparison(probs):
    if 'P(A|B)_Bayes' not in probs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Bar comparison
    ax1 = axes[0]
    labels = ['P(Fallo)\nBase', 'P(Fallo|Evidencia)\nBayes']
    values = [probs['P(A)'], probs['P(A|B)_Bayes']]
    colors = ['#7c3aed', '#06b6d4']
    bars = ax1.bar(labels, values, color=colors, width=0.5, edgecolor='none', alpha=0.9)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='white', fontfamily='monospace')
    ax1.set_ylim(0, max(values) * 1.3)
    ax1.set_title('Comparación de Probabilidades', pad=10)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylabel('Probabilidad')

    # Probability update visualization
    ax2 = axes[1]
    stages = ['Prior\nP(A)', 'Verosimilitud\nP(B|A)', 'Evidencia\nP(B)', 'Posterior\nP(A|B)']
    vals = [probs['P(A)'], probs.get('P(B|A)', 0), probs.get('P(B)', 0), probs['P(A|B)_Bayes']]
    colors2 = ['#8b5cf6', '#06b6d4', '#f59e0b', '#10b981']
    for i, (s, v, c) in enumerate(zip(stages, vals, colors2)):
        ax2.barh(i, v, color=c, alpha=0.85, edgecolor='none', height=0.6)
        ax2.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9,
                fontweight='bold', color='white', fontfamily='monospace')
    ax2.set_yticks(range(len(stages)))
    ax2.set_yticklabels(stages, fontsize=8)
    ax2.set_title('Actualización Bayesiana', pad=10)
    ax2.set_xlabel('Probabilidad')
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.set_xlim(0, max(vals) * 1.4)

    fig.tight_layout()
    return fig

def plot_confusion_matrix(nb_results):
    if not nb_results:
        return None
    cm = nb_results['cm']
    fig, ax = plt.subplots(figsize=(6, 5))
    
    mask_colors = np.array([[0.1, 0.4], [0.6, 0.9]])
    if cm.shape == (2,2):
        color_data = mask_colors
    else:
        color_data = cm / cm.max()

    im = ax.imshow(color_data, cmap=plt.cm.get_cmap('RdPu'), vmin=0, vmax=1)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = ['TN', 'FP', 'FN', 'TP'][(i*2)+j] if cm.shape == (2,2) else ''
            ax.text(j, i, f'{cm[i,j]}\n{label}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', fontfamily='monospace')

    classes = nb_results['classes']
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([f'Pred: {c}' for c in classes], fontsize=9)
    ax.set_yticklabels([f'Real: {c}' for c in classes], fontsize=9)
    ax.set_title('Matriz de Confusión', pad=10)
    fig.tight_layout()
    return fig

# ─── MAIN APP ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-hero">
    <h1>🌸 APP Probabilidades</h1>
    <p>ANÁLISIS BAYESIANO · DETECCIÓN DE ANOMALÍAS · NAIVE BAYES CLASSIFIER</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Cargar Datos")
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])

    if uploaded_file:
        df = load_csv(uploaded_file)
        numeric_cols, categorical_cols, datetime_cols, binary_cols = detect_columns(df)

        st.markdown("---")
        st.markdown("### 🎯 Variable Objetivo")
        all_binary = binary_cols + [c for c in numeric_cols if df[c].nunique() <= 2]
        target_options = list(set(all_binary)) if all_binary else df.columns.tolist()
        target_col = st.selectbox("Selecciona variable objetivo (fallo/anomalía)", target_options)

        st.markdown("---")
        st.markdown("### 🔍 Variable Evidencia")
        evidence_options = [c for c in numeric_cols if c != target_col]
        if evidence_options:
            evidence_col = st.selectbox("Variable para P(B|A)", evidence_options)
            col_data = df[evidence_col].dropna()
            default_thresh = float(col_data.quantile(0.75))
            threshold = st.slider(
                f"Umbral para '{evidence_col}'",
                float(col_data.min()), float(col_data.max()),
                default_thresh,
                step=float((col_data.max()-col_data.min())/100)
            )
        else:
            evidence_col = None
            threshold = None

        st.markdown("---")
        st.markdown("### 🤖 Features para Naive Bayes")
        feature_cols = st.multiselect(
            "Selecciona variables predictoras",
            [c for c in numeric_cols + binary_cols if c != target_col],
            default=[c for c in numeric_cols + binary_cols if c != target_col][:4]
        )

        run_analysis = st.button("⚡ EJECUTAR ANÁLISIS")

# ─── MAIN CONTENT ────────────────────────────────────────────────────────────────

if not uploaded_file:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; opacity: 0.4;">
        <div style="font-size: 5rem; margin-bottom: 20px;">📊</div>
        <p style="font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #64748b; letter-spacing: 0.1em;">
            CARGA UN ARCHIVO CSV PARA COMENZAR EL ANÁLISIS
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = load_csv(uploaded_file)
numeric_cols, categorical_cols, datetime_cols, binary_cols = detect_columns(df)

# ─── DATA OVERVIEW ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>📋 Resumen del Dataset</h2></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[0]:,}</div><div class="metric-label">Registros</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Variables</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{df.isnull().sum().sum()}</div><div class="metric-label">Valores Nulos</div></div>', unsafe_allow_html=True)
with col4:
    mem = df.memory_usage(deep=True).sum() / 1024
    st.markdown(f'<div class="metric-card"><div class="metric-value">{mem:.1f}KB</div><div class="metric-label">Tamaño</div></div>', unsafe_allow_html=True)

st.markdown("**Tipo de columnas detectadas:**")
tags_html = ""
for c in numeric_cols:
    tags_html += f'<span class="tag numeric">📈 {c}</span>'
for c in categorical_cols:
    tags_html += f'<span class="tag categorical">🏷️ {c}</span>'
for c in datetime_cols:
    tags_html += f'<span class="tag datetime">📅 {c}</span>'
for c in binary_cols:
    tags_html += f'<span class="tag binary">⚡ {c}</span>'
st.markdown(tags_html, unsafe_allow_html=True)

with st.expander("📄 Vista previa del dataset"):
    st.dataframe(df.head(20), use_container_width=True)

if 'target_col' not in dir():
    st.info("👈 Configura los parámetros en el panel lateral para continuar.")
    st.stop()

# ─── ANALYSIS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header"><h2>🧮 Análisis Probabilístico Bayesiano</h2></div>', unsafe_allow_html=True)

probs = compute_probabilities(df, target_col, evidence_col, threshold)

if probs is None:
    st.error("La variable objetivo debe ser binaria (2 valores únicos). Selecciona otra columna.")
    st.stop()

# Probability display
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("P(A)", probs.get('P(A)', 0), "Prob. de Fallo"),
    ("P(¬A)", probs.get('P(no_A)', 0), "Prob. No Fallo"),
    ("P(B|A)", probs.get('P(B|A)', None), "Verosimilitud"),
    ("P(A|B)", probs.get('P(A|B)_Bayes', None), "Posterior Bayes"),
]
for col, (label, val, desc) in zip([c1, c2, c3, c4], metrics):
    with col:
        if val is not None:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val:.4f}</div><div class="metric-label">{label} — {desc}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card" style="opacity:0.4"><div class="metric-value">—</div><div class="metric-label">{label} — selecciona evidencia</div></div>', unsafe_allow_html=True)

# Bayes formula display
if 'P(A|B)_Bayes' in probs:
    st.markdown(f"""
    <div class="insight-box" style="text-align:center; font-size:1rem;">
        <div class="insight-header">Teorema de Bayes Aplicado</div>
        P(Fallo | {evidence_col} &gt; {threshold:.2f}) = 
        <span style="color:#06b6d4">{probs['P(B|A)']:.4f}</span> × 
        <span style="color:#7c3aed">{probs['P(A)']:.4f}</span> / 
        <span style="color:#f59e0b">{probs['P(B)']:.4f}</span> = 
        <span style="color:#10b981; font-size:1.3rem; font-weight:700">{probs['P(A|B)_Bayes']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

# ─── NAIVE BAYES ─────────────────────────────────────────────────────────────────
nb_results = None
if feature_cols and len(feature_cols) >= 1:
    nb_results = run_naive_bayes(df, target_col, feature_cols)

if nb_results:
    st.markdown('<div class="section-header"><h2>🤖 Clasificador Naive Bayes</h2></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nb_results["acc"]:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nb_results["sensitivity"]:.1%}</div><div class="metric-label">Sensibilidad (Recall)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nb_results["specificity"]:.1%}</div><div class="metric-label">Especificidad</div></div>', unsafe_allow_html=True)

# ─── INSIGHTS ────────────────────────────────────────────────────────────────────
insights = generate_insights(df, target_col, probs, nb_results)
st.markdown('<div class="section-header"><h2>💡 Insights Estadísticos</h2></div>', unsafe_allow_html=True)
for insight in insights:
    st.markdown(f'<div class="insight-box"><div class="insight-header">insight</div>{insight}</div>', unsafe_allow_html=True)

# ─── VISUALIZATIONS ──────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header"><h2>📊 Visualizaciones</h2></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Histogramas", "📅 Serie Temporal", "🎯 Probabilidades", "🔲 Matriz Confusión"])

with tab1:
    fig = plot_histograms(df, numeric_cols, target_col)
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No hay columnas numéricas suficientes para graficar.")

with tab2:
    fig = plot_temporal(df, datetime_cols, target_col)
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No se detectaron columnas de fecha/tiempo en el dataset.")

with tab3:
    fig = plot_posterior_comparison(probs)
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Selecciona una variable de evidencia para ver la comparación de probabilidades.")

with tab4:
    if nb_results:
        col_cm, col_metrics = st.columns([1, 1])
        with col_cm:
            fig = plot_confusion_matrix(nb_results)
            if fig:
                st.pyplot(fig, use_container_width=True)
        with col_metrics:
            st.markdown("""
            <div class="insight-box">
            <div class="insight-header">Interpretación de la Matriz</div>
            <b style='color:#10b981'>VP (Verdaderos Positivos / TP):</b> Fallos detectados correctamente<br><br>
            <b style='color:#ef4444'>FN (Falsos Negativos):</b> Fallos NO detectados (peligroso)<br><br>
            <b style='color:#f59e0b'>FP (Falsos Positivos):</b> Alarmas falsas<br><br>
            <b style='color:#06b6d4'>VN (Verdaderos Negativos / TN):</b> No-fallos correctamente clasificados
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Selecciona al menos una feature en el panel lateral para entrenar el clasificador.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; font-family:'DM Mono',monospace; font-size:0.7rem; color:#e8d5f0; letter-spacing:0.1em;">
APP PROBABILIDADES · TEOREMA DE BAYES · NAIVE BAYES CLASSIFIER 🌸
</p>
""", unsafe_allow_html=True)