-- HOLO-STYLE DASHBOARD SCHEMA (v3) — Ops Expansion
-- Run in Supabase SQL editor

create extension if not exists "pgcrypto";

-- =========================
-- 1) TABLES
-- =========================

-- Holdings Snapshots
create table if not exists holdings_snapshots (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  as_of_date date not null,
  total_value_usd numeric(18, 2),
  holdings_json jsonb not null,      -- { "TQQQ": {"shares": 100, "price": 50.0, "value": 5000}, ... }
  alloc_json jsonb,                 -- { "TQQQ": 0.62, "SPLG": 0.18, ... }
  target_alloc_json jsonb,          -- { "TQQQ": 0.60, "SGOV": 0.20, ... }
  notes text,
  user_id uuid default auth.uid()
);

-- Signals
create table if not exists signals (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  as_of_date date not null,
  regime text not null,             -- 'ON' or 'OFF'
  confirmed boolean default false,
  confirm_day int,
  reason_json jsonb,                -- { "rule": "Ensemble", "inputs": {...}, "explain": "..." }
  target_alloc_json jsonb,
  user_id uuid default auth.uid()
);

-- News Items
create table if not exists news_items (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  ts timestamptz not null,
  source text not null,
  title text not null,
  url text,
  tickers text[],
  tags text[],
  summary text,
  user_id uuid default auth.uid()
);

-- Reports
create table if not exists reports (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  title text not null,
  type text not null,               -- 'md', 'csv', 'png'
  storage_path text not null,
  tags text[],
  notes text,
  user_id uuid default auth.uid()
);

-- Audit Log
create table if not exists audit_log (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  actor uuid default auth.uid(),
  action text not null,
  payload_json jsonb,
  user_id uuid default auth.uid()
);

-- [NEW] Trigger Evaluations (Daily Singleton)
create table if not exists trigger_evaluations (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  as_of_date date not null,
  inputs_json jsonb not null,       -- { "close": 500, "ma200": 480, ... }
  checks_json jsonb not null,       -- { "price_gt_ma": true, "env_ok": true, ... }
  trigger_ok boolean not null,
  notes text,
  user_id uuid default auth.uid()
);

-- [NEW] Execution Checklists (Daily Singleton)
create table if not exists execution_checklists (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  as_of_date date not null,
  pretrade_json jsonb not null,     -- { "fx_rate": 1400, "plan": {...}, "broker_checked": true }
  pretrade_complete boolean default false,
  posttrade_json jsonb not null,    -- { "executed": {...}, "fees": 50, ... }
  executed_complete boolean default false,
  override_used boolean default false, -- If true, ignore complete flags in gate
  user_id uuid default auth.uid()
);

-- [NEW] Decision Logs (Daily Singleton for Overrides)
create table if not exists decision_logs (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  as_of_date date not null,
  memo text,
  override_reason text,             -- Required if overriding
  user_id uuid default auth.uid()
);

-- [NEW] Decision Attachments
create table if not exists decision_attachments (
  id uuid default gen_random_uuid() primary key,
  created_at timestamptz default now(),
  as_of_date date not null,
  storage_path text not null,       -- "user_id/as_of_date/filename.png"
  file_name text,
  mime_type text,
  user_id uuid default auth.uid()
);

-- =========================
-- 2) CONSTRAINTS
-- =========================
do $$
begin
  -- Regimes
  if not exists (select 1 from pg_constraint where conname = 'signals_regime_check') then
    alter table signals add constraint signals_regime_check check (regime in ('ON','OFF'));
  end if;

  -- [NEW] Daily Unique Constraints per User
  if not exists (select 1 from pg_constraint where conname = 'uniq_trigger_user_date') then
    alter table trigger_evaluations add constraint uniq_trigger_user_date unique (user_id, as_of_date);
  end if;
  
  if not exists (select 1 from pg_constraint where conname = 'uniq_checklist_user_date') then
    alter table execution_checklists add constraint uniq_checklist_user_date unique (user_id, as_of_date);
  end if;

  if not exists (select 1 from pg_constraint where conname = 'uniq_decision_user_date') then
    alter table decision_logs add constraint uniq_decision_user_date unique (user_id, as_of_date);
  end if;

end $$;

-- =========================
-- 3) INDEXES
-- =========================
create index if not exists holdings_snapshots_user_date_idx on holdings_snapshots(user_id, as_of_date desc);
create index if not exists signals_user_date_idx on signals(user_id, as_of_date desc);
create index if not exists news_items_user_ts_idx on news_items(user_id, ts desc);
create index if not exists reports_user_created_idx on reports(user_id, created_at desc);
create index if not exists audit_log_user_created_idx on audit_log(user_id, created_at desc);

-- [NEW] Indexes
create index if not exists trigger_user_date_idx on trigger_evaluations(user_id, as_of_date desc);
create index if not exists checklist_user_date_idx on execution_checklists(user_id, as_of_date desc);
create index if not exists decision_user_date_idx on decision_logs(user_id, as_of_date desc);
create index if not exists attachments_user_date_idx on decision_attachments(user_id, as_of_date desc);

-- =========================
-- 4) RLS ENABLE
-- =========================
alter table holdings_snapshots enable row level security;
alter table signals enable row level security;
alter table news_items enable row level security;
alter table reports enable row level security;
alter table audit_log enable row level security;
-- [NEW]
alter table trigger_evaluations enable row level security;
alter table execution_checklists enable row level security;
alter table decision_logs enable row level security;
alter table decision_attachments enable row level security;

-- =========================
-- 5) POLICIES (Owner Access Only)
-- =========================
-- Helper to perform bulk policy drop/create
do $$
declare
  t text;
begin
  foreach t in array array['holdings_snapshots', 'signals', 'news_items', 'reports', 'audit_log', 'trigger_evaluations', 'execution_checklists', 'decision_logs', 'decision_attachments']
  loop
    -- Drop old
    execute format('drop policy if exists "%1$s_select_own" on %1$s', t);
    execute format('drop policy if exists "%1$s_insert_own" on %1$s', t);
    execute format('drop policy if exists "%1$s_update_own" on %1$s', t);
    execute format('drop policy if exists "%1$s_delete_own" on %1$s', t);
    
    -- Create new
    execute format('create policy "%1$s_select_own" on %1$s for select using (user_id = auth.uid())', t);
    execute format('create policy "%1$s_insert_own" on %1$s for insert with check (user_id = auth.uid())', t);
    execute format('create policy "%1$s_update_own" on %1$s for update using (user_id = auth.uid()) with check (user_id = auth.uid())', t);
    execute format('create policy "%1$s_delete_own" on %1$s for delete using (user_id = auth.uid())', t);
  end loop;
end $$;
