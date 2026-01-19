import streamlit as st
import pandas as pd
import plotly.express as px
from db_supabase import SupabaseDB
from predict import LeadScorer
from orchestrator import LeadOrchestrator
import joblib
import os
import io

# Page Config
st.set_page_config(page_title="Lead Scoring AI - Automatización", layout="wide")

# Initialize Components
@st.cache_resource
def load_components():
    db = None
    try:
        db = SupabaseDB()
    except Exception as e:
        st.sidebar.error(f"Error de conexión: {str(e)}")
        st.warning("Supabase no conectado. Algunos paneles estarán vacíos.")
    
    scorer = LeadScorer()
    orchestrator = LeadOrchestrator()
    return db, scorer, orchestrator

db, scorer, orchestrator = load_components()

# Database Connection Helper in Sidebar
with st.sidebar:
    st.header("🔗 Conexión")
    if not db:
        st.error("Supabase no conectado")
        with st.expander("¿Cómo conectar?"):
            st.write("Añade estas variables en 'Secrets' de Streamlit Cloud o en tu `.env` local:")
            st.code("SUPABASE_URL=tu_url\nSUPABASE_KEY=tu_anon_key")
            st.write("[Consigue tus credenciales aquí](https://supabase.com/dashboard/project/moxiotloytrnlnfgyvdw/settings/api)")
            st.info("⚠️ **IMPORTANTE**: Asegúrate de añadir el esquema 'leadscoring' en la configuración de la API de Supabase.")
    else:
        st.success("Conectado a Supabase (Esquema: leadscoring)")

# Title
st.title("🎯 Lead Scoring AI - Automatización")
st.markdown("Sistema de predicción automática de calidad de leads.")

# Sidebar - Model Info & Sync
with st.sidebar:
    st.header("⚙️ Estado del Sistema")
    if scorer.pipeline:
        st.success("Modelo Activo: LogisticRegression")
        if os.path.exists('model_metadata.pkl'):
            meta = joblib.load('model_metadata.pkl')
            st.metric("Precisión (AUC)", f"{meta['auc']:.4f}")
    
    st.divider()
    st.header("🔄 Acciones")
    if st.button("Sincronizar Pendientes", help="Procesa los leads de Supabase que aún no tienen score"):
        with st.spinner("Procesando leads pendientes..."):
            count = orchestrator.sync_unscored_leads()
            st.success(f"¡Sincronización completa! Se han procesado {count} leads.")
            st.rerun()

# Main Tabs
tab1, tab2 = st.tabs(["📊 Panel de Resultados", "📤 Procesamiento por Lote (CSV)"])

with tab1:
    st.header("Histórico de Scoring (Automático)")
    
    if db:
        try:
            # Fetch scores using the isolated schema
            response = db.client.schema("leadscoring").table("lead_scores").select("*, leads(raw_data)").execute()
            data = response.data
            if data:
                # Process data for display
                display_data = []
                for item in data:
                    raw = item.get('leads', {}).get('raw_data', {})
                    display_data.append({
                        "Fecha": item['created_at'],
                        "Canal": raw.get('channel', 'N/A'),
                        "Campaña": raw.get('campaign', 'N/A'),
                        "Score": item['score'],
                        "Probabilidad": f"{item['probability']*100:.1f}%",
                        "Factores Positivos": ", ".join(item['explanation'].get('top_positive_factors', []))
                    })
                
                df_scores = pd.DataFrame(display_data)
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Leads", len(df_scores))
                c2.metric("Media Score", round(df_scores['Score'].mean(), 1))
                c3.metric("Leads Hot (>70)", len(df_scores[df_scores['Score'] > 70]))

                st.dataframe(df_scores.sort_values("Fecha", ascending=False), use_container_width=True)
                
                # Visuals
                fig = px.box(df_scores, x="Canal", y="Score", color="Canal", title="Calidad de Leads por Canal")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay leads procesados. Sube un archivo o pulsa 'Sincronizar Pendientes'.")
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
    else:
        st.error("Conexión a base de datos necesaria para el panel histórico.")

with tab2:
    st.header("Subir archivo de Leads")
    st.write("Sube un CSV para que el sistema limpie, normalice y prediga automáticamente cada lead.")
    
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Vista previa del archivo:")
        st.dataframe(df_upload.head())
        
        if st.button("🚀 Iniciar Procesamiento Automático"):
            with st.spinner("Limpiando, normalizando y prediciendo..."):
                # Use orchestrator to process
                temp_path = "temp_batch.csv"
                df_upload.to_csv(temp_path, index=False)
                results = orchestrator.process_batch(temp_path)
                
                st.success(f"¡Éxito! Se han procesado {len(results)} leads automáticamente.")
                # Show summary
                scores = [r['score'] for r in results if r]
                st.write(f"Score promedio del lote: **{sum(scores)/len(scores):.1f}**")
                
                if os.path.exists(temp_path): os.remove(temp_path)
                st.rerun()
