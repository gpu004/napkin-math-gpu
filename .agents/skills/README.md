# Agent skills

Vendored from [cursor/plugins](https://github.com/cursor/plugins) at `1dcc2ba` (MIT, Copyright (c) 2026 Lauren Tan for `pstack`). Edited only to point generated skills at `.agents/skills/` instead of `.cursor/skills/`.

| Skill | Source |
| --- | --- |
| create-verification-skill (+ references/feature-map-example), maintain-verification-skill | `pstack/skills` |
| no-comments (+ references/comment-sicko.md) | [backnotprop/pstack](https://github.com/backnotprop/pstack/tree/02b59d7f73a693d40ad2090d00789f3e3a772151/skills/no-comments) at `02b59d7` (MIT, Copyright (c) 2026 Lauren Tan), unedited |

All three run on request only: `/create-verification-skill` generates `.agents/skills/verify-<app>/`, `/maintain-verification-skill` audits it, and `/no-comments` runs the Comment Sicko comment audit over a diff and fixes the findings it accepts.

`no-comments` refers to pstack skills that are not vendored here: `poteto-mode`, `how`, `why`, `architect`, `principle-fix-root-causes`, and `principle-redesign-from-first-principles`. Without them, the agent runs those steps from their names alone. Comment Sicko's prompt is in `references/comment-sicko.md`, so it works in any harness that can spawn a subagent.
