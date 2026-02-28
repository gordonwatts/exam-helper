---
name: github-issue-workflow
description: Manage GitHub issue execution with `gh` in this repository, including scanning open issues, retrieving issue details, creating issue branches, opening draft PRs, and keeping test/push hygiene. Use when a user asks to work an issue, check issue/PR status, or start/continue issue-driven implementation.
---

# Github Issue Workflow

## Overview

Use `gh` to discover and inspect issues, then follow the repository's required issue-development flow.
Prefer deterministic commands and keep work aligned to branch and PR policy.

## Scan And Inspect Issues

- List open issues:
  - `gh issue list --state open --limit 200`
- View one issue:
  - `gh issue view <issue-number> --json number,title,body,state,labels,assignees,url`
- View one issue with discussion:
  - `gh issue view <issue-number> --comments`

If `gh` is unavailable, ask the user to install/authenticate `gh`, then retry the same command.

## Command Playbook

Run these in order for a new issue implementation.

1. Sync `main` first:
   - `git checkout main`
   - `git pull --ff-only origin main`
2. Inspect and confirm target issue:
   - `gh issue view <issue-number> --json number,title,body,state,labels,assignees,url`
   - `gh issue view <issue-number> --comments`
3. Create issue branch before edits:
   - `git checkout -b issue-<issue-number>-<short-slug>`
4. Implement changes and run tests:
   - `uv run --extra dev pytest -q`
5. Commit and push branch:
   - `git add -A`
   - `git commit -m "<focused-change-message>"`
   - `git push -u origin issue-<issue-number>-<short-slug>`
6. Open draft PR to `main` with auto-close reference:
   - `gh pr create --base main --head issue-<issue-number>-<short-slug> --draft --title "<pr-title>" --body "fix #<issue-number>"`

## Start New Issue Checklist

1. Confirm clean context:
   - Check current branch and status with `git status -sb`.
2. Scan open issues:
   - Run `gh issue list --state open --limit 200`.
3. Load issue details:
   - Run both `gh issue view <n> --json ...` and `gh issue view <n> --comments`.
4. Update `main` before branch work:
   - Run `git checkout main` and `git pull --ff-only origin main`.
5. Create issue branch before any edits:
   - Use `git checkout -b issue-<n>-<slug>`.
6. Start implementation and open draft PR early:
   - Push and create draft PR as soon as meaningful implementation starts.

## Resume Existing Issue Checklist

1. Verify issue branch exists and switch to it:
   - `git checkout issue-<issue-number>-<short-slug>`
2. Refresh issue/PR context:
   - `gh issue view <issue-number> --comments`
   - `gh pr status`
3. Sync from `main` if needed (merge/rebase per repo preference).
4. Continue implementation on issue branch only.
5. Run tests before push:
   - `uv run --extra dev pytest -q`
6. Push updates and keep PR draft until ready:
   - `git push`
   - Mark PR ready only after tests/review items are complete.

## Policy Guardrails

- Do not implement issue work directly on `main`.
- Do not switch tools if a workflow command fails due to sandbox restrictions; rerun with proper escalation.
- Keep commits focused and descriptive.
- Avoid destructive git commands unless explicitly requested.
- Include issue-closing reference in PR description (for example `fix #123`).
- If sandbox/network blocks required update commands, request elevated execution and retry.

## Reference

See `references/agents-github-issue-workflow.md` for the extracted source policy text from this repository's `AGENTS.md`.
