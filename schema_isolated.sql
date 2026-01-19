-- Create a new schema to isolate lead scoring tables
CREATE SCHEMA IF NOT EXISTS leadscoring;

-- Enable UUID extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Leads Table: Stores the raw lead information
CREATE TABLE IF NOT EXISTS leadscoring.leads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id TEXT, 
    raw_data JSONB NOT NULL 
);

-- 2. Processing Features Table: Stores features used for model training/inference
CREATE TABLE IF NOT EXISTS leadscoring.lead_features (
    lead_id UUID REFERENCES leadscoring.leads(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    features_json JSONB NOT NULL, 
    PRIMARY KEY (lead_id, created_at)
);

-- 3. Model Versions: Tracks different models
CREATE TABLE IF NOT EXISTS leadscoring.model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    algorithm TEXT NOT NULL, 
    metrics JSONB, 
    artifact_path TEXT, 
    active BOOLEAN DEFAULT FALSE
);

-- 4. Lead Scores: The result of the model
CREATE TABLE IF NOT EXISTS leadscoring.lead_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lead_id UUID REFERENCES leadscoring.leads(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version_id UUID REFERENCES leadscoring.model_versions(id),
    score INTEGER CHECK (score >= 0 AND score <= 100), 
    probability FLOAT, 
    explanation JSONB 
);

-- Indexing for speed
CREATE INDEX IF NOT EXISTS idx_leads_raw_data ON leadscoring.leads USING gin (raw_data);
CREATE INDEX IF NOT EXISTS idx_lead_scores_lead_id ON leadscoring.lead_scores(lead_id);
CREATE INDEX IF NOT EXISTS idx_lead_scores_created_at ON leadscoring.lead_scores(created_at DESC);

-- Note: To access this from the Supabase client (PostgREST), 
-- you must add 'leadscoring' to the 'db_schemas' setting in the Supabase Dashboard
-- or use the schema prefix if configured to expose multiple schemas.
