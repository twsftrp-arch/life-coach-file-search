---
name: handoff-status
description: Create or refresh a concise repo-local SESSION-HANDOFF checkpoint.
---

# Handoff Status

Use this skill before idle, compact, session switch, or cross-machine transfer.

## Target

Prefer:

- `docs/SESSION-HANDOFF.md`

Fallback:

- `SESSION-HANDOFF.md`

## Content

Write a concise checkpoint with:

- date and machine if relevant
- branch
- dirty status
- completed work
- verification evidence
- gated actions not performed
- next safe step

Do not include secrets or private token values.

