-- Optional: enables POST /api/v1/sentiment/callback to find rows by Azure Video Indexer video id.
-- Run once on your Supabase project if you use VI webhooks.

ALTER TABLE public.sentiment_outputs
  ADD COLUMN IF NOT EXISTS azure_video_indexer_id text NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_outputs_azure_vi
  ON public.sentiment_outputs (azure_video_indexer_id)
  WHERE azure_video_indexer_id IS NOT NULL;
