import os
from supabase import create_client, Client
from dotenv import load_dotenv
import json
from datetime import datetime

# Load env variables (assumes .env file is present)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class SupabaseDB:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def insert_lead(self, raw_data: dict, tenant_id: str = "default"):
        """Inserts a new lead and returns its ID."""
        data = {
            "raw_data": raw_data,
            "tenant_id": tenant_id
        }
        response = self.client.table("leads").insert(data).execute()
        return response.data[0]['id']

    def insert_features(self, lead_id: str, features: dict):
        """Inserts processed features for a lead."""
        data = {
            "lead_id": lead_id,
            "features_json": features
        }
        self.client.table("lead_features").insert(data).execute()

    def insert_score(self, lead_id: str, score: int, probability: float, explanation: dict, model_version_id: str = None):
        """Inserts the calculated score."""
        data = {
            "lead_id": lead_id,
            "score": score,
            "probability": probability,
            "explanation": explanation,
            "model_version_id": model_version_id
        }
        self.client.table("lead_scores").insert(data).execute()

    def get_lead_history(self, lead_id: str):
        """Retrieves scoring history for a lead."""
        response = self.client.table("lead_scores")\
            .select("*")\
            .eq("lead_id", lead_id)\
            .order("created_at", desc=True)\
            .execute()
        return response.data

    def register_model_version(self, algorithm: str, metrics: dict, artifact_path: str):
        """Registers a new model version."""
        data = {
            "algorithm": algorithm,
            "metrics": metrics,
            "artifact_path": artifact_path,
            "active": True 
        }
        # Ideally disable previous active models here
        return self.client.table("model_versions").insert(data).execute()

    def get_unscored_leads(self):
        """Fetches leads from the 'leads' table that don't have an entry in 'lead_scores' yet."""
        # We perform a join or a filter. In Supabase client, we can use 'not.in' or similar, 
        # but a simple way is to select leads and then check scores.
        # For simplicity in this logic, we'll use a subquery style or just fetch recent ones.
        # Better: select leads left join lead_scores where score is null.
        # Since supabase-py has limitations with complex joins, we'll use an RPC or a view if needed, 
        # but here we'll do a simple fetch of recent leads.
        response = self.client.table("leads").select("*, lead_scores(id)").execute()
        # Filter leads where 'lead_scores' is an empty list
        unscored = [item for item in response.data if not item.get('lead_scores')]
        return unscored

if __name__ == "__main__":
    # Test connection (will fail if no env vars)
    try:
        db = SupabaseDB()
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Supabase init failed: {e}")
