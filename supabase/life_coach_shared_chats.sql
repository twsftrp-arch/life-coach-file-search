create extension if not exists pgcrypto;

create table if not exists public.life_coach_shared_chats (
  id uuid primary key default gen_random_uuid(),
  share_token text not null unique,
  source_session_id uuid references public.life_coach_sessions(id) on delete set null,
  source_session_key text not null,
  owner_user_id uuid not null references auth.users(id) on delete cascade,
  title text not null default '공유된 대화',
  messages jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  revoked_at timestamptz,
  constraint life_coach_shared_chats_share_token_format
    check (share_token ~ '^sh_[A-Za-z0-9_-]{24,96}$'),
  constraint life_coach_shared_chats_messages_array
    check (jsonb_typeof(messages) = 'array')
);

create index if not exists life_coach_shared_chats_owner_updated_idx
  on public.life_coach_shared_chats (owner_user_id, updated_at desc);

create index if not exists life_coach_shared_chats_source_idx
  on public.life_coach_shared_chats (source_session_key, owner_user_id, updated_at desc);

create index if not exists life_coach_shared_chats_active_token_idx
  on public.life_coach_shared_chats (share_token)
  where revoked_at is null;

create or replace function public.set_life_coach_shared_chats_updated_at()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists life_coach_shared_chats_set_updated_at
  on public.life_coach_shared_chats;

create trigger life_coach_shared_chats_set_updated_at
  before update on public.life_coach_shared_chats
  for each row execute function public.set_life_coach_shared_chats_updated_at();

alter table public.life_coach_shared_chats enable row level security;
revoke all on table public.life_coach_shared_chats from anon, authenticated;
grant all on table public.life_coach_shared_chats to service_role;
