import os
from supabase import create_client, Client
from dotenv import load_dotenv
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables (assumes .env file is present)
load_dotenv()

class SupabaseDB:
    def __init__(self):
        # Dynamic check to support Streamlit Secrets and late-loaded env vars
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        # Fallback to Streamlit Secrets if available
        if not self.url or not self.key:
            try:
                import streamlit as st
                self.url = st.secrets.get("SUPABASE_URL")
                self.key = st.secrets.get("SUPABASE_KEY")
            except:
                pass
                
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL o SUPABASE_KEY no configuradas. Añádelas a Secrets en Streamlit Cloud o a tu archivo .env local.")
            
        # Basic key validation (Project keys are JWT, sbp_ keys are management)
        if self.key.startswith("sbp_"):
            raise ValueError("⚠️ Estás usando una 'Management Key' (sbp_...). Necesitas la 'Anon Key' o 'Service Role Key' de Project Settings -> API.")

        try:
            self.client: Client = create_client(self.url, self.key)
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {e}")
            raise ConnectionError(f"No se pudo inicializar el cliente de Supabase: {e}")

    def _handle_response(self, func, *args, **kwargs):
        """Helper to handle database responses and errors."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "JWT" in err_msg:
                raise PermissionError("Error 401: API Key inválida o expirada. Revisa tus credenciales.")
            elif "404" in err_msg:
                raise FileNotFoundError("Error 404: Tabla o esquema no encontrado. ¿Has creado el esquema 'leadscoring'?")
            else:
                raise Exception(f"Error en base de datos: {err_msg}")

    def insert_lead(self, raw_data: dict, tenant_id: str = "default"):
        """Inserts a new lead and returns its ID."""
        data = {"raw_data": raw_data, "tenant_id": tenant_id}
        def call():
            return self.client.schema("leadscoring").table("leads").insert(data).execute()
        response = self._handle_response(call)
        return response.data[0]['id']

    def insert_features(self, lead_id: str, features: dict):
        """Inserts processed features for a lead."""
        data = {"lead_id": lead_id, "features_json": features}
        def call():
            return self.client.schema("leadscoring").table("lead_features").insert(data).execute()
        self._handle_response(call)

    def insert_score(self, lead_id: str, score: int, probability: float, explanation: dict, model_version_id: str = None):
        """Inserts the calculated score."""
        data = {
            "lead_id": lead_id,
            "score": score,
            "probability": probability,
            "explanation": explanation,
            "model_version_id": model_version_id
        }
        def call():
            return self.client.schema("leadscoring").table("lead_scores").insert(data).execute()
        self._handle_response(call)

    def get_lead_history(self, lead_id: str):
        """Retrieves scoring history for a lead."""
        def call():
            return self.client.schema("leadscoring").table("lead_scores")\
                .select("*")\
                .eq("lead_id", lead_id)\
                .order("created_at", desc=True)\
                .execute()
        response = self._handle_response(call)
        return response.data

    def register_model_version(self, algorithm: str, metrics: dict, artifact_path: str):
        """Registers a new model version."""
        data = {
            "algorithm": algorithm,
            "metrics": metrics,
            "artifact_path": artifact_path,
            "active": True 
        }
        def call():
            return self.client.schema("leadscoring").table("model_versions").insert(data).execute()
        return self._handle_response(call)

    def get_unscored_leads(self):
        """Fetches leads from the 'leads' table that don't have an entry in 'lead_scores' yet."""
        def call():
            return self.client.schema("leadscoring").table("leads").select("*, lead_scores(id)").execute()
        response = self._handle_response(call)
        unscored = [item for item in response.data if not item.get('lead_scores')]
        return unscored

if __name__ == "__main__":
    try:
        db = SupabaseDB()
        print("Conexión exitosa.")
    except Exception as e:
        print(f"Error: {e}")
