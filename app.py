import streamlit as st
import pandas as pd
import joblib

# 1. Configuración inicial de la página
st.set_page_config(page_title="Clasificador E-shop Premium", page_icon="🛍️", layout="wide")

st.title("🛍️ Simulador de Navegación: Predicción de Clics Premium")
st.markdown("""
Esta aplicación utiliza el modelo de Machine Learning (Random Forest) desarrollado para predecir 
si un usuario interactuará con un producto de la categoría **Premium** en nuestro e-shop.
""")

# 2. Mostrar Resultados del Modelo (Requisito de rúbrica)
st.header("📊 Rendimiento del Modelo (Solemne 1)")
col1, col2, col3 = st.columns(3)
col1.metric("Exactitud (Accuracy)", "92.8%")
col2.metric("Sensibilidad (Recall)", "93.0%")
col3.metric("Precisión (Precision)", "91.5%")
st.markdown("---")

# 3. Simulador Interactivo
st.header("🎛️ Simulador Interactivo")
st.markdown("Ingresa los parámetros de navegación para predecir el comportamiento del cliente:")

# Cargar el modelo pre-entrenado
try:
    modelo = joblib.load('modelo_rf_premium.pkl')
except FileNotFoundError:
    st.error("Por favor, asegúrate de que el archivo 'modelo_rf_premium.pkl' esté en la misma carpeta.")
    st.stop()

# Configurar controles (widgets) en barras laterales y columnas
st.sidebar.header("Parámetros del Clic")

# Variables base
page_1 = st.sidebar.selectbox("Categoría Principal (1: Pantalones, 2: Faldas, 3: Blusas, 4: Ofertas)", [1, 2, 3, 4])
location = st.sidebar.slider("Ubicación en la foto (1 a 6)", 1, 6, 1)
colour = st.sidebar.slider("Color (1 a 14)", 1, 14, 1)
model_photography = st.sidebar.radio("Fotografía del Modelo", [1, 2], horizontal=True)
page = st.sidebar.slider("Profundidad de Página (1 a 5)", 1, 5, 1)

# Variables Feature Engineering
st.sidebar.header("Patrón de Comportamiento")
is_weekend = st.sidebar.radio("¿Es Fin de Semana?", [0, 1], horizontal=True)
session_length = st.sidebar.number_input("Total de clics en la sesión", min_value=1, max_value=50, value=10)
orden_clic_actual = st.sidebar.number_input("Número del clic actual", min_value=1, max_value=50, value=5)
prev_category = st.sidebar.selectbox("Categoría del clic anterior (0 si es el primero)", [0, 1, 2, 3, 4])

# Cálculo automático de variables compuestas
cat_page_interact = page_1 * page
order_ratio = orden_clic_actual / session_length

# Botón para predecir
if st.button("🚀 Predecir Intención de Clic", type="primary"):
    # Crear dataframe con el input del usuario asegurando el mismo orden que X_fe
    input_data = pd.DataFrame([[
        location, colour, page_1, model_photography, page, 
        is_weekend, session_length, cat_page_interact, prev_category, order_ratio
    ]], columns=['location', 'colour', 'page_1_(main_category)', 'model_photography', 'page',
                 'is_weekend', 'session_length', 'cat_page_interact', 'prev_category', 'order_ratio'])
    
    # Hacer predicción
    prediccion = modelo.predict(input_data)[0]
    probabilidad = modelo.predict_proba(input_data)[0][1] * 100
    
    # Mostrar el resultado visualmente
    st.markdown("### Resultado de la Predicción")
    if prediccion == 1:
        st.success(f"🌟 **ALTA PROBABILIDAD (Premium)**: El usuario tiene perfil para productos de alto margen. Probabilidad: {probabilidad:.1f}%")
        st.balloons()
    else:
        st.info(f"🛒 **COMPRA ESTÁNDAR**: El usuario busca el catálogo normal. Probabilidad de ser Premium: {probabilidad:.1f}%")