import streamlit as st
import pandas as pd
import plotly.express as px
from db_supabase import SupabaseDB
from predict import LeadScorer
import joblib
import os

# Page Config
st.set_page_config(page_title="Lead Scoring Dashboard", layout="wide")

# Initialize DB and Scorer
@st.cache_resource
def load_components():
    db = None
    try:
        db = SupabaseDB()
    except Exception as e:
        st.warning("Supabase not connected. Running in demo mode.")
    
    scorer = LeadScorer()
    return db, scorer

db, scorer = load_components()

# Title
st.title("🎯 Lead Scoring AI - Dashboard")
st.markdown("Visualiza y predice la calidad de tus leads en tiempo real.")

# Sidebar - Model Info
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    if scorer.pipeline:
        st.success("Modelo Cargado: LogisticRegression")
        if os.path.exists('model_metadata.pkl'):
            meta = joblib.load('model_metadata.pkl')
            st.metric("ROC-AUC", f"{meta['auc']:.4f}")
    else:
        st.error("Modelo no encontrado. Por favor entrena el modelo primero.")

# Main Tabs
tab1, tab2 = st.tabs(["📊 Dashboard de Leads", "🔍 Predictor Interactivo"])

with tab1:
    st.header("Histórico de Scoring")
    
    # In a real scenario, fetch from Supabase. For demo, we might use the CSV.
    if db:
        try:
            # Simplification: Fetching all lead scores
            response = db.client.table("lead_scores").select("*, leads(raw_data)").execute()
            data = response.data
            if data:
                df_scores = pd.DataFrame(data)
                st.dataframe(df_scores)
                
                # Plot distribution
                fig = px.histogram(df_scores, x="score", nbins=20, title="Distribución de Scores")
                st.plotly_chart(fig)
            else:
                st.info("No hay datos en Supabase todavía. Procesa leads con el orquestador para ver resultados aquí.")
        except Exception as e:
            st.error(f"Error al cargar datos de Supabase: {e}")
    else:
        st.info("Conecta Supabase para ver el histórico real. Mostrando datos locales de 'marketing_data.csv' como referencia:")
        if os.path.exists('marketing_data.csv'):
            df_local = pd.read_csv('marketing_data.csv')
            st.dataframe(df_local.head(10))

with tab2:
    st.header("Predecir Nuevo Lead")
    
    if scorer.pipeline:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Datos del Lead")
            channel = st.selectbox("Canal", ["Organic Search", "Paid Social", "Email", "Referral", "Direct"])
            campaign = st.selectbox("Campaña", ["Summer_Sale", "Black_Friday", "Webinar_Q1", "Demo_Request", "None"])
            time_on_site = st.slider("Tiempo en el sitio (seg)", 0, 1800, 300)
            pages_visited = st.number_input("Páginas visitadas", 1, 50, 3)
            newsletter_sub = st.checkbox("Suscrito al Newsletter")
            downloads = st.number_input("Descargas realizadas", 0, 10, 0)
            
            if st.button("Calcular Score"):
                lead_data = {
                    "channel": channel,
                    "campaign": campaign,
                    "time_on_site": time_on_site,
                    "pages_visited": pages_visited,
                    "newsletter_sub": 1 if newsletter_sub else 0,
                    "downloads": downloads
                }
                
                result = scorer.predict(lead_data)
                
                with col2:
                    st.subheader("Resultado")
                    score = result['score']
                    
                    if score >= 70:
                        st.balloons()
                        st.success(f"🔥 **Lead de Alta Prioridad: {score}/100**")
                    elif score >= 40:
                        st.warning(f"⚡ **Lead con Potencial: {score}/100**")
                    else:
                        st.error(f"❄️ **Lead de Baja Prioridad: {score}/100**")
                    
                    # Probability gauge (simulation)
                    st.metric("Probabilidad de Conversión", f"{result['probability']*100:.1f}%")
                    
                    st.subheader("¿Por qué este score?")
                    st.write("**Factores Positivos:**")
                    for factor in result['explanation']['top_positive_factors']:
                        st.write(f"- ✅ {factor.replace('_', ' ').title()}")
                        
                    st.write("**Factores Negativos:**")
                    for factor in result['explanation']['top_negative_factors']:
                        st.write(f"- ❌ {factor.replace('_', ' ').title()}")
    else:
        st.warning("El predictor no está listo. Asegúrate de que 'lead_scoring_model.pkl' existe.")
