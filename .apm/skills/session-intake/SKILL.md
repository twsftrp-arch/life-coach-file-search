---
name: session-intake
description: Run the mandatory repo intake before work, handoff recovery, or cross-machine continuation.
---

# Session Intake

Use this skill before changing or judging a repository.

## Steps

1. Find repo root:
   - `git rev-parse --show-toplevel`
2. Report branch and status:
   - `git status -sb`
3. Read local instruction files if present:
   - `AGENTS.md`
   - `CLAUDE.md`
   - `GEMINI.md`
   - `docs/SESSION-HANDOFF.md`
   - `SESSION-HANDOFF.md`
   - relevant `docs/ai/*`
4. If handoff is missing, search:
   - `rg --files -g 'SESSION-HANDOFF.md' -g 'HANDOFF.md'`
5. Only then proceed or report that intake is complete.

## Output

Keep the report short:

- repo
- branch
- dirty summary
- files read
- immediate blocker, if any

