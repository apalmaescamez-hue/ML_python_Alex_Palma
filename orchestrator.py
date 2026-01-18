import pandas as pd
import logging
from db_supabase import SupabaseDB
from predict import LeadScorer
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeadOrchestrator:
    def __init__(self, action_threshold=70):
        self.scorer = LeadScorer()
        self.db = None
        self.action_threshold = action_threshold
        
        try:
            self.db = SupabaseDB()
            logger.info("Orchestrator: Supabase connection initialized.")
        except Exception as e:
            logger.warning(f"Orchestrator: Supabase connection failed ({e}). Running in local-only mode.")

    def process_new_lead(self, lead_data: dict, tenant_id: str = "default"):
        """
        Processes a single lead: Ingest -> Clean/Normalize/Predict -> Save Results -> Action
        """
        logger.info(f"Processing new lead from channel: {lead_data.get('channel', 'unknown')}")

        # 1. Ingest into Supabase (Raw data)
        lead_id = None
        if self.db:
            try:
                lead_id = self.db.insert_lead(lead_data, tenant_id)
                logger.info(f"Lead ingested with ID: {lead_id}")
            except Exception as e:
                logger.error(f"Failed to ingest lead: {e}")

        # 2. Predict (This handles cleaning & normalization internally via the Pipeline)
        # Note: 'predict' method in LeadScorer uses the joblib pipeline which contains:
        # SimpleImputer (Cleaning) + StandardScaler (Normalization) + OneHot (Encoding)
        result = self.scorer.predict(lead_data)
        
        if not result:
            logger.error("Prediction failed.")
            return None

        score = result['score']
        logger.info(f"Lead Score: {score}")

        # 3. Save Score and Explanation to Supabase
        if self.db and lead_id:
            try:
                self.db.insert_score(
                    lead_id=lead_id,
                    score=score,
                    probability=result['probability'],
                    explanation=result['explanation']
                )
                logger.info("Score persisted to Supabase.")
            except Exception as e:
                logger.error(f"Failed to save score: {e}")

        # 4. "Action" Logic (Replacing n8n conditional steps)
        if score >= self.action_threshold:
            self._trigger_high_intent_action(lead_id, score, result['explanation'])
        
        return result

    def process_batch(self, file_path: str):
        """Processes a CSV file as a batch of leads."""
        logger.info(f"Starting batch process for {file_path}")
        df = pd.read_csv(file_path)
        
        results = []
        for index, row in df.iterrows():
            # Convert row to dict, excluding target if present
            lead_dict = row.to_dict()
            if 'converted' in lead_dict: del lead_dict['converted']
            if 'lead_id' in lead_dict: del lead_dict['lead_id']
            
            res = self.process_new_lead(lead_dict)
            results.append(res)
            
        logger.info(f"Batch processing complete. Processed {len(results)} leads.")
        return results

    def sync_unscored_leads(self):
        """Fetches leads from Supabase that don't have scores and processes them."""
        logger.info("Syncing unscored leads from Supabase...")
        if not self.db:
            logger.error("DB not connected. Cannot sync.")
            return 0
        
        unscored = self.db.get_unscored_leads()
        logger.info(f"Found {len(unscored)} unscored leads.")
        
        count = 0
        for lead in unscored:
            # The 'raw_data' contains the features
            lead_data = lead['raw_data']
            lead_id = lead['id']
            
            # Predict
            result = self.scorer.predict(lead_data)
            if result:
                self.db.insert_score(
                    lead_id=lead_id,
                    score=result['score'],
                    probability=result['probability'],
                    explanation=result['explanation']
                )
                
                # Check for action
                if result['score'] >= self.action_threshold:
                    self._trigger_high_intent_action(lead_id, result['score'], result['explanation'])
                
                count += 1
        
        logger.info(f"Sync complete. Processed {count} leads.")
        return count

    def _trigger_high_intent_action(self, lead_id, score, explanation):
        """Simulates an action like sending an email or Slack alert."""
        logger.info(f"🔥 ACTION TRIGGERED: High intent lead detected! Score: {score}")
        logger.info(f"Reasoning: {explanation['top_positive_factors']}")
        # In a real scenario, here you would call a Slack/Discord webhook or Email API.

if __name__ == "__main__":
    # Example usage
    orchestrator = LeadOrchestrator(action_threshold=70)
    
    # Simulate a high-intent lead
    high_intent_lead = {
        "channel": "Email",
        "campaign": "Demo_Request",
        "time_on_site": 600,
        "pages_visited": 10,
        "newsletter_sub": 1,
        "downloads": 3
    }
    
    orchestrator.process_new_lead(high_intent_lead)
