import joblib
import pandas as pd
import numpy as np

class LeadScorer:
    def __init__(self, model_path='lead_scoring_model.pkl', metadata_path='model_metadata.pkl'):
        try:
            self.pipeline = joblib.load(model_path)
            self.metadata = joblib.load(metadata_path)
            self.model = self.pipeline.named_steps['classifier']
            self.preprocessor = self.pipeline.named_steps['preprocessor']
        except Exception as e:
            print(f"Error loading model: {e}")
            self.pipeline = None

    def predict(self, lead_data: dict):
        """
        Receives a dictionary of lead data, returns score and explanation.
        """
        if self.pipeline is None:
            return None
        
        # Convert dict to DataFrame
        df_lead = pd.DataFrame([lead_data])
        
        # Ensure correct column order/existence (fill missing with NaN)
        all_cols = self.metadata['num_cols'] + self.metadata['cat_cols']
        for col in all_cols:
            if col not in df_lead.columns:
                df_lead[col] = np.nan
        
        df_lead = df_lead[all_cols]
        
        # Probabilities
        prob = self.pipeline.predict_proba(df_lead)[0][1]
        score = round(prob * 100)
        
        # Explainability (for Logistic Regression)
        explanation = self.get_explanation(df_lead)
        
        return {
            "score": score,
            "probability": round(float(prob), 4),
            "explanation": explanation
        }

    def get_explanation(self, df_lead):
        """Extracts top features contributing to the score."""
        # Get feature names after preprocessing
        feature_names = self._get_feature_names()
        
        # Transform lead
        transformed_x = self.preprocessor.transform(df_lead)
        if hasattr(transformed_x, 'toarray'):
            transformed_x = transformed_x.toarray()
            
        # Get coefficients
        coefs = self.model.coef_[0]
        
        # Calculate impact (value * weight)
        impacts = transformed_x[0] * coefs
        
        # Map to original features (simplified for baseline)
        impact_series = pd.Series(impacts, index=feature_names)
        
        top_positive = impact_series.sort_values(ascending=False).head(3).index.tolist()
        top_negative = impact_series.sort_values(ascending=True).head(3).index.tolist()
        
        return {
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative
        }

    def _get_feature_names(self):
        """Helper to get feature names from ColumnTransformer."""
        output_features = []
        for name, pipe, features in self.preprocessor.transformers_:
            if name == 'remainder': continue
            if hasattr(pipe, 'named_steps'):
                if 'onehot' in pipe.named_steps:
                    # Categorical with OneHot
                    categories = pipe.named_steps['onehot'].get_feature_names_out(features)
                    output_features.extend(categories)
                else:
                    # Numerical
                    output_features.extend(features)
            else:
                 output_features.extend(features)
        return output_features

if __name__ == "__main__":
    scorer = LeadScorer()
    sample_lead = {
        "channel": "Email",
        "campaign": "Demo_Request",
        "time_on_site": 300,
        "pages_visited": 5,
        "newsletter_sub": 1,
        "downloads": 2
    }
    result = scorer.predict(sample_lead)
    print(f"Prediction Result: {result}")
