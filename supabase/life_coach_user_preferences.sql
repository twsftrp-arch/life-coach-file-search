create table if not exists public.life_coach_user_preferences (
  user_id uuid primary key references auth.users(id) on delete cascade,
  coaching_style text not null default 'balanced',
  custom_instructions text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint life_coach_user_preferences_style_check
    check (coaching_style in ('balanced', 'gentle', 'direct', 'accountability', 'analytical')),
  constraint life_coach_user_preferences_custom_length
    check (char_length(custom_instructions) <= 1200)
);

create or replace function public.set_life_coach_user_preferences_updated_at()
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

drop trigger if exists life_coach_user_preferences_set_updated_at
  on public.life_coach_user_preferences;

create trigger life_coach_user_preferences_set_updated_at
  before update on public.life_coach_user_preferences
  for each row execute function public.set_life_coach_user_preferences_updated_at();

alter table public.life_coach_user_preferences enable row level security;
revoke all on table public.life_coach_user_preferences from anon, authenticated;
grant all on table public.life_coach_user_preferences to service_role;
