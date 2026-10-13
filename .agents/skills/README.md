# Agent skills

Vendored from [cursor/plugins](https://github.com/cursor/plugins) at `1dcc2ba` (MIT, Copyright (c) 2026 Lauren Tan for `pstack`). Edited only to point generated skills at `.agents/skills/` instead of `.cursor/skills/`.

| Skill | Source |
| --- | --- |
| create-verification-skill (+ references/feature-map-example), maintain-verification-skill | `pstack/skills` |

Both run on request only: `/create-verification-skill` generates `.agents/skills/verify-<app>/`, `/maintain-verification-skill` audits it.
