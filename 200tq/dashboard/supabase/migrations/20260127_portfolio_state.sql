CREATE TABLE IF NOT EXISTS portfolio_state (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL DEFAULT 'default',
  tqqq_shares INTEGER NOT NULL DEFAULT 0,
  sgov_shares INTEGER NOT NULL DEFAULT 0,
  last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  source TEXT NOT NULL DEFAULT 'manual',
  
  CONSTRAINT portfolio_state_user_unique UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_state_user ON portfolio_state(user_id);

INSERT INTO portfolio_state (user_id, tqqq_shares, sgov_shares, source)
VALUES ('default', 0, 0, 'manual')
ON CONFLICT (user_id) DO NOTHING;
